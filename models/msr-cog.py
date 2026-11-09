import torch
from torch import nn
from models.lightmedseg import LightMedSeg


class RegressionHead(nn.Module):
    def __init__(self):
        super().__init__()

        self.mlp = ...


class MSRCog(nn.Module):
    def __init__(self, seg_weights: str, input_size: tuple = (1, 192, 192, 192)):
        super().__init__()

        self.input_size = input_size
        C, D, H, W = input_size

        self.bn_channels = 3  # no. of bottleneck input channels

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

        # Whether the segmentation backbone donwsampled the input
        # before passing it to the encoder
        ds_enabled = self.seg_backbone._parameters["downsample"]

        # If downsampling, calculate embedding size
        if ds_enabled:
            D /= 2
            H /= 2
            W /= 2

        # Fused segmentation bottleneck output size
        D_bn = D / 4
        H_bn = H / 4
        W_bn = W / 4

        self.bn_size = (D_bn, H_bn, W_bn)

        reg_in_size = D_bn * H_bn * W_bn * self.bn_channels
        hid_size = reg_in_size // 2
        out_size = 2

        self.reg_head = nn.Sequential(
            nn.Linear(reg_in_size, hid_size),
            nn.SiLU(),
            nn.Linear(hid_size, out_size)
        )
