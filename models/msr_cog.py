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
        input_size: tuple = (1, 192, 192, 192),
        demo_size=14,
        out_size=1,
        mgen_anchors=256,
        mgen_stage_channels=(8, 16, 32, 64),
        mgen_downsample=False,
        seg_anchors=8,
        seg_stage_channels=(8, 16, 32, 64),
        seg_downsample=False,
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

        self._hyperparams = {
            "input_size": input_size,
            "demo_size": demo_size,
            "out_size": out_size,
            "mgen_anchors": mgen_anchors,
            "mgen_stage_channels": mgen_stage_channels,
            "mgen_downsample": mgen_downsample,
            "seg_anchors": seg_anchors,
            "seg_stage_channels": seg_stage_channels,
            "seg_downample": seg_downsample
        }

        self.input_size = input_size
        C, D, H, W = input_size

        # Initiate mask generator module
        self.mask_gen = LightMedSeg(
            n_classes=2,
            in_channels=1,
            num_anchors=mgen_anchors,
            stage_channels=mgen_stage_channels,
            downsample=mgen_downsample,
            metadata_film=False,
        )

        # Initialize the segmentation backbone
        self.seg_backbone = LightMedSeg(
            n_classes=4,  # background, GM, WM, CSF,
            in_channels=1,
            num_anchors=seg_anchors,
            stage_channels=seg_stage_channels,
            downsample=seg_downsample,
            metadata_film=False,
            boundary_refine=False,
        )

        # Freeze the segmentation backbone

        # Add channel counts for skip_3, skip_4 and e4_out
        bn_channels = seg_stage_channels[2] + 2 * seg_stage_channels[3]

        # Reduced channel count of bottneleck input into regression head
        rd_channels = bn_channels // 16

        self.demo_film = MetadataFiLM(
            in_channels=bn_channels, meta_dim=demo_size, hidden_dim=128
        )

        # Whether the segmentation backbone donwsampled the input
        # before passing it to the encoder

        # If downsampling, calculate embedding size
        if seg_downsample:
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
            nn.SiLU(),
        )

        self.reg_head = RegressionHead(
            in_size=reg_in_size, hid_size=hid_size, out_size=out_size
        )

    @staticmethod
    def from_seg(
        path,
        device=None,
        input_size=None,
        demo_size=None,
        out_size=None,
        mgen_anchors=None,
        mgen_stage_channels=None,
        mgen_downsample=None,
    ):
        """
        Loads model using pretrained segmentation backbone weights.
        Sets backbone hyperparameters and loads the state dict.
        """
        checkpoint = torch.load(path, map_location=device)
        hparams = checkpoint["hyperparams"]
        model = MSRCog(
            input_size=input_size,
            demo_size=demo_size,
            out_size=out_size,
            mgen_anchors=mgen_anchors,
            mgen_stage_channels=mgen_stage_channels,
            mgen_downsample=mgen_downsample,
            seg_anchors=hparams["num_anchors"],
            seg_stage_channels=hparams["stage_channels"],
            seg_downample=hparams["downsample"]
        )
        
        model.seg_backbone.load_state_dict(checkpoint["model_state_dict"])
        
        return model
    
    @staticmethod
    def load(path, device=None):
        checkpoint = torch.load(path, map_location=device)
        hparams = checkpoint["hyperparams"]
        model = MSRCog(
            input_size=hparams["input_size"],
            demo_size=hparams["demo_size"],
            out_size=hparams["out_size"],
            mgen_anchors=hparams["mgen_anchors"],
            mgen_stage_channels=hparams["mgen_stage_channels"],
            mgen_downsample=hparams["mgen_downsample"],
            seg_anchors=hparams["seg_anchors"],
            seg_stage_channels=hparams["seg_stage_channels"],
            seg_downsample=hparams["seg_downsample"],
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        return model

    def save(self, path, metadata=None):
        save_dict = {
            "metadata": metadata,
            "model_state_dict": self.state_dict(),
            "hyperparams": self.hyperparams(),
        }

        torch.save(save_dict, path)

    def hyperparams(self):
        return self._hyperparams

    def freeze_seg(self):
        self.seg_backbone.requires_grad_(False)

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
