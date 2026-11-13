from typing import Any


import torch
from torch import Tensor, nn
import torch.nn.functional as F
from models.lightmedseg import LightMedSeg
from models.modules import MetadataFiLM


class RegressionHead(nn.Module):
    def __init__(self, in_size, out_size, hid_size=None):
        super().__init__()

        if hid_size == None:
            hid_size = in_size // 2

        self.mlp = nn.Sequential(
            nn.Linear(in_size, hid_size), nn.SiLU(), nn.Linear(hid_size, out_size)
        )

    def forward(self, X):
        return self.mlp(X)


class MSRCog(nn.Module):
    def __init__(
        self,
        seg_weights: str,
        input_size: tuple = (1, 192, 192, 192),
        demo_size=14,
        out_size=1,
    ):
        """
        input_size: `tuple[int, int, int, int]`
            Input image size in the format `(channel_count, depth, width, height)`.
        demo_size: `int`
            Size (length) of the demographics vector.
        out_size: `int`
            Output vector size. Set to 2 for cogscore + disagnosis. 1 by default.
        """
        super().__init__()

        self.input_size = input_size
        C, D, H, W = input_size

        # Initiate mask generator module
        self.mask_gen = LightMedSeg(
            n_classes=2,
            in_channels=1,
            num_anchors=256,
            stage_channels=(8, 16, 32, 64),
            metadata_film=False,
        )

        # Load the pretrained segmentation backbone
        self.seg_backbone = LightMedSeg.load(seg_weights)
        # Freeze the segmentation backbone
        self.seg_backbone.requires_grad_(False)

        stage_channels = self.seg_backbone._hyperparams["stage_channels"]

        # Add channel counts for skip_3, skip_4 and e4_out
        bn_channels = stage_channels[2] + 2 * stage_channels[3]
        
        # Reduced channel count of bottneleck input into regression head
        rd_channels = bn_channels // 16
        

        self.demo_film = MetadataFiLM(
            in_channels=bn_channels, meta_dim=demo_size, hidden_dim=128
        )

        # Whether the segmentation backbone donwsampled the input
        # before passing it to the encoder
        ds_enabled = self.seg_backbone._hyperparams["downsample"]

        # If downsampling, calculate embedding size
        if ds_enabled:
            D //= 2
            H //= 2
            W //= 2

        # Fused segmentation bottleneck output size
        D_bn = D // 8
        H_bn = H // 8
        W_bn = W // 8

        # bn_size: tuple = (D_bn, H_bn, W_bn)

        reg_in_size = D_bn * H_bn * W_bn * rd_channels
        hid_size = reg_in_size // 2
        
        self.bn_reducer = nn.Sequential(
            nn.Conv3d(bn_channels, rd_channels, 1),
            nn.BatchNorm3d(rd_channels),
            nn.SiLU()
        )

        self.reg_head = RegressionHead(
            in_size=reg_in_size, hid_size=hid_size, out_size=out_size
        )

    def fuse_bottleneck(self, e4_out, skip_2, skip_1, target_size) -> Tensor:
        """
        Fuses skip connections and bottleneck (encoder 4) output from segmentation backbone for regression.
        """
        bottleneck = F.interpolate(
            e4_out, target_size, mode="trilinear", align_corners=False
        )
        skip_2 = F.interpolate(
            skip_2, target_size, mode="trilinear", align_corners=False
        )
        skip_1 = F.interpolate(
            skip_1, target_size, mode="trilinear", align_corners=False
        )
        
        # Concatenate along channel dimension
        return torch.cat([bottleneck, skip_2, skip_1], dim=1)

    def forward(self, X, demographics):        
        mask = torch.softmax(self.mask_gen(X), dim=1)
        
        mask = mask[:, 0:1, :, :, :]
                
        masked = torch.mul(mask, X)

        e4_out, skip_2, skip_1 = self.seg_backbone.forward_bottleneck(masked)

        _, _, D, H, W = e4_out.size()

        fused = self.fuse_bottleneck(e4_out, skip_2, skip_1, target_size=(D, H, W))

        conditioned = self.demo_film(fused, demographics)
        
        reduced = self.bn_reducer(conditioned)
        
        flattened = reduced.flatten(start_dim=1)

        out = self.reg_head(flattened)

        return out
