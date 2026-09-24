import torch
import torch.nn as nn
import torch.nn.functional as F


class TextureMap(nn.Module):
    def __init__(self, in_channels=8):
        super().__init__()

        kernel = 5
        stride = 1
        padding = 2

        num_groups = 4

        self.conv1 = nn.Sequential(
            nn.Conv3d(
                in_channels, in_channels, kernel, stride, padding, groups=in_channels
            ),
            nn.GroupNorm(num_groups, in_channels),
            nn.SiLU(),
        )

        self.conv2 = nn.Conv3d(in_channels, 1, 1, 1, 0)

        self.sigmoid = nn.Sigmoid()

    def forward(self, f0):
        smoothed = self.conv1(f0)
        resid = torch.abs(smoothed - f0)
        out = self.sigmoid(self.conv2(resid))
        return out


class SpatialGatingHead(nn.Module):
    def __init__(self, in_channels=8):
        super().__init__()

        c1_kernel = 3
        c1_stride = 1
        c1_padding = 1

        c2_kernel = 3
        c2_stride = 1
        c2_padding = 1

        n_groups = 4

        self.conv1 = nn.Sequential(
            nn.Conv3d(
                in_channels,
                2 * in_channels,
                c1_kernel,
                c1_stride,
                c1_padding,
                groups=in_channels,
            ),
            nn.GroupNorm(n_groups, 2 * in_channels),
            nn.SiLU(),
        )

        self.conv2 = nn.Sequential(
            nn.Conv3d(2 * in_channels, 1, c2_kernel, c2_stride, c2_padding),
            nn.Sigmoid(),
        )

        self.projection = nn.Sequential(
            nn.Conv3d(in_channels=1, out_channels=2, kernel_size=1, stride=1, padding=0),
            nn.Softmax(dim=1)
        )

    def forward(self, f0):
        x1 = self.conv1(f0)
        x2 = self.conv2(x1)
        alphas = self.projection(x2)
        
        return alphas[:, 0:1, ...], alphas[:, 1:2, ...]

class AdaptiveFeatureMixer(nn.Module):
    def __init__(self, in_channels=8):
        super().__init__()
        
        self.conv1 = nn.Conv3d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.conv2 = nn.Conv3d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        
        
    def forward(self, f0, a1, a2):
        z1 = self.conv1(f0)
        z2 = self.conv2(f0)
        out = a1 * z1 + a2 * z2
        return out


class LSPM(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.texture_map = TextureMap()
        self.spatial_gating = SpatialGatingHead()
        self.feature_mixer = AdaptiveFeatureMixer()

    def forward(self, f0):
        T = self.texture_map(f0)
        a1, a2 = self.spatial_gating(f0)
        out = self.feature_mixer(f0, a1, a2)
        
        return T, out
