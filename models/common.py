from torch.nn import Module
import torch.nn as nn
import torch.nn.functional as F
import torch

class GhostConv3D(Module):
    def __init__(self, in_channels, out_channels=8, downscale=True, ratio=2):
        super().__init__()
        self.out_channels = out_channels
        int_channels = out_channels // ratio
        new_channels = int_channels * (ratio - 1)

        pr_kernel_size = 3
        pr_stride = 2 if downscale else 1
        pr_padding = 1

        dw_kernel_size = 3
        dw_stride = 1
        dw_padding = 1

        # n. of groups for group norm
        n_groups = 4

        self.primary = nn.Sequential(
            nn.Conv3d(
                in_channels,
                int_channels,
                pr_kernel_size,
                pr_stride,
                pr_padding,
                bias=False,
            ),
            nn.GELU(),
        )

        self.depthwise = nn.Sequential(
            nn.Conv3d(
                int_channels,
                new_channels,
                dw_kernel_size,
                dw_stride,
                dw_padding,
                groups=int_channels,
                bias=False,
            ),
            nn.GELU(),
        )

        self.group_norm = nn.GroupNorm(n_groups, out_channels)

    def forward(self, X):
        x1 = self.primary(X)
        x2 = self.depthwise(x1)
        cat = torch.cat([x1, x2], dim=1)
        out = self.group_norm(cat)
        return out


class GlobalAnchorDetector(Module):
    def __init__(self, in_channels, out_channels, num_anchors=8):
        super().__init__()

        self.num_anchors = num_anchors

        kernel = 3
        stride = 2
        padding = 1

        # n. of groups for GN
        n_groups = 4

        # mlp hidden layer size
        hidden_width = 256

        self.phi = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel, stride, padding),
            nn.GroupNorm(n_groups, out_channels),
            nn.SiLU(),
        )

        self.pool = nn.AdaptiveAvgPool3d([1, 1, 1])

        self.mlp = nn.Sequential(
            nn.Linear(out_channels, hidden_width),
            nn.SiLU(),
            nn.Linear(hidden_width, 3 * num_anchors),
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        batch_size = x.shape[0]

        x1 = self.phi(x)
        x2 = self.pool(x1)
        x2 = x2.flatten(start_dim=1)
        x3 = self.mlp(x2)
        logits = self.sigmoid(x3)
        S = logits.reshape(batch_size, self.num_anchors, 3)
        return S


class SpatialAnchorFiLM(Module):
    def __init__(self, in_channels, num_anchors=8):
        super().__init__()

        self.gamma_proj = nn.Linear(num_anchors * 3, in_channels)
        self.beta_proj = nn.Linear(num_anchors * 3, in_channels)

    def forward(self, S, gconvout):

        s = S.flatten(start_dim=1)
        gamma = self.gamma_proj(s)
        beta = self.beta_proj(s)

        gamma = gamma[:, :, None, None, None]
        beta = beta[:, :, None, None, None]

        F_cond = (gamma + 1) * gconvout + beta
        return F_cond


class MetadataFiLM(nn.Module):
    def __init__(self, in_channels, meta_dim=4, hidden_dim=128):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(meta_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 2 * in_channels),
        )

        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, x, metadata):
        gamma, beta = self.mlp(metadata).chunk(2, dim=1)

        gamma = gamma[:, :, None, None, None]
        beta = beta[:, :, None, None, None]

        return (gamma + 1) * x + beta


class SEBlock(Module):
    def __init__(self, in_channels):
        super().__init__()

        reduction_ratio = max(16, in_channels // 2)

        self.bottleneck = nn.Sequential(
            nn.AdaptiveAvgPool3d((1, 1, 1)),
            nn.Flatten(start_dim=1),
            nn.Linear(in_channels, in_channels // reduction_ratio),
            nn.SiLU(),
            nn.Linear(in_channels // reduction_ratio, in_channels),
            nn.Sigmoid(),
        )

    def forward(self, input):
        z = self.bottleneck(input)
        z = z[:, :, None, None, None]
        
        out = z * input
        return out


class Encoder(Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        num_anchors=8,
        metadata_film=True,
        downsample=True,
    ):
        super().__init__()

        self.metadata_film = metadata_film

        self.ghost_conv = GhostConv3D(in_channels, out_channels, downscale=False)
        self.spatial_film = SpatialAnchorFiLM(out_channels, num_anchors=num_anchors)

        if metadata_film:
            self.meta_film = MetadataFiLM(out_channels)

        self.detail_preserve = nn.Sequential(
            nn.Conv3d(out_channels, out_channels, 3, 1, 1, groups=out_channels),
            nn.GroupNorm(4, out_channels),
            nn.SiLU(),
        )

        self.smoothing = nn.Conv3d(out_channels, out_channels, 1, 1, 0)

        self.se = SEBlock(out_channels)

        self.max_pool = nn.MaxPool3d(2, 2) if downsample else nn.Identity()

    def forward(self, f0_prime, S, T, metadata):
        convout = self.ghost_conv(f0_prime)
        conditioned = self.spatial_film(S, convout)

        if self.metadata_film:
            conditioned = self.meta_film(conditioned, metadata)

        stage_resolution = conditioned.shape[2:]
        T_resized = F.interpolate(T, size=stage_resolution, mode="trilinear")

        z_detail = self.detail_preserve(conditioned)
        z_smooth = self.smoothing(conditioned)

        texture_aware = T_resized * z_detail + (1 - T_resized) * z_smooth

        E_i = self.se(texture_aware)

        out = self.max_pool(E_i)
        return out, E_i
    

class MultiScaleSkipFusion(Module):
    def __init__(self, stage_channels):
        super().__init__()

        internal_channels = 64
        concat_channels = 256
        n_stages = 4

        self.proj_1 = nn.Conv3d(
            in_channels=stage_channels[0],
            out_channels=internal_channels,
            kernel_size=1,
            stride=1,
            padding=0,
        )
        self.proj_2 = nn.Conv3d(
            in_channels=stage_channels[1],
            out_channels=internal_channels,
            kernel_size=1,
            stride=1,
            padding=0,
        )
        self.proj_3 = nn.Conv3d(
            in_channels=stage_channels[2],
            out_channels=internal_channels,
            kernel_size=1,
            stride=1,
            padding=0,
        )
        self.proj_4 = nn.Conv3d(
            in_channels=stage_channels[3],
            out_channels=internal_channels,
            kernel_size=1,
            stride=1,
            padding=0,
        )

        self.controller_1 = nn.Sequential(
            nn.Conv3d(
                in_channels=concat_channels,
                out_channels=internal_channels,
                kernel_size=1,
                stride=1,
                padding=0,
            ),
            nn.SiLU(),
            nn.Conv3d(
                in_channels=internal_channels,
                out_channels=n_stages,
                kernel_size=1,
                stride=1,
                padding=0,
            ),
        )

        self.controller_2 = nn.Sequential(
            nn.Conv3d(
                in_channels=concat_channels,
                out_channels=internal_channels,
                kernel_size=1,
                stride=1,
                padding=0,
            ),
            nn.SiLU(),
            nn.Conv3d(
                in_channels=internal_channels,
                out_channels=n_stages,
                kernel_size=1,
                stride=1,
                padding=0,
            ),
        )

        self.out_proj_1 = nn.Conv3d(
            in_channels=internal_channels,
            out_channels=stage_channels[3],
            kernel_size=1,
            stride=1,
            padding=0,
        )
        self.out_proj_2 = nn.Conv3d(
            in_channels=internal_channels,
            out_channels=stage_channels[2],
            kernel_size=1,
            stride=1,
            padding=0,
        )
        self.out_proj_3 = nn.Conv3d(
            in_channels=internal_channels,
            out_channels=stage_channels[1],
            kernel_size=1,
            stride=1,
            padding=0,
        )
        self.out_proj_4 = nn.Conv3d(
            in_channels=internal_channels,
            out_channels=stage_channels[0],
            kernel_size=1,
            stride=1,
            padding=0,
        )

    def forward(self, E_1, E_2, E_3, E_4):


        size_4 = E_1.shape[2:]
        size_3 = E_2.shape[2:]
        size_2 = E_3.shape[2:]
        size_1 = E_4.shape[2:]

        E_hat_1 = self.proj_1(E_1)
        E_hat_2 = self.proj_2(E_2)
        E_hat_3 = self.proj_3(E_3)
        E_hat_4 = self.proj_4(E_4)

        skip_4 = self.out_proj_4(E_hat_2)
        skip_4 = F.interpolate(skip_4, size=size_4, mode="trilinear")

        skip_3 = self.out_proj_3(E_hat_1)
        skip_3 = F.interpolate(skip_3, size=size_3, mode="trilinear")

        # resizing encoder outputs for skip 2
        up_2_1 = F.interpolate(E_hat_1, size=size_2, mode="trilinear")
        up_2_2 = F.interpolate(E_hat_2, size=size_2, mode="trilinear")
        up_2_3 = F.interpolate(E_hat_3, size=size_2, mode="trilinear")
        up_2_4 = F.interpolate(E_hat_4, size=size_2, mode="trilinear")

        cat_2 = torch.cat([up_2_1, up_2_2, up_2_3, up_2_4], dim=1)
        logits_2 = self.controller_2(cat_2)
        weights_2 = F.softmax(logits_2, dim=1)

        skip_2 = (
            weights_2[:, 0:1, :, :, :] * up_2_1
            + weights_2[:, 1:2, :, :, :] * up_2_2
            + weights_2[:, 2:3, :, :, :] * up_2_3
            + weights_2[:, 3:4, :, :, :] * up_2_4
        )

        skip_2 = self.out_proj_2(skip_2)

        # resizing encoder outputs for skip 1
        up_1_1 = F.interpolate(E_hat_1, size=size_1, mode="trilinear")
        up_1_2 = F.interpolate(E_hat_2, size=size_1, mode="trilinear")
        up_1_3 = F.interpolate(E_hat_3, size=size_1, mode="trilinear")
        up_1_4 = F.interpolate(E_hat_4, size=size_1, mode="trilinear")

        cat_1 = torch.cat([up_1_1, up_1_2, up_1_3, up_1_4], dim=1)
        logits_1 = self.controller_1(cat_1)
        weights_1 = F.softmax(logits_1, dim=1)

        skip_1 = (
            weights_1[:, 0:1, :, :, :] * up_1_1
            + weights_1[:, 1:2, :, :, :] * up_1_2
            + weights_1[:, 2:3, :, :, :] * up_1_3
            + weights_1[:, 3:4, :, :, :] * up_1_4
        )

        skip_1 = self.out_proj_1(skip_1)

        return skip_1, skip_2, skip_3, skip_4


class Decoder(Module):
    def __init__(self, in_channels, out_channels, num_anchors=8, upsample=True):
        super().__init__()

        self.num_branches = 3

        self.upsample_conv = (
            nn.ConvTranspose3d(in_channels, in_channels, kernel_size=2, stride=2)
            if upsample
            else nn.Conv3d(in_channels, in_channels, kernel_size=1, stride=1)
        )
        self.spb_conv = nn.Conv3d(
            in_channels=num_anchors,
            out_channels=in_channels,
            kernel_size=1,
            stride=1,
            padding=0,
        )

        self.fuse_conv = nn.Conv3d(
            in_channels=2 * in_channels,
            out_channels=in_channels,
            kernel_size=1,
            stride=1,
            padding=0,
        )

        self.blending = nn.Sequential(
            nn.Conv3d(
                in_channels=in_channels,
                out_channels=self.num_branches,
                kernel_size=1,
                stride=1,
                padding=0,
            ),
            nn.Softmax(dim=1),
        )

        self.f1 = nn.Sequential(
            nn.Conv3d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                groups=out_channels,
            ),
            nn.GroupNorm(4, out_channels),
            nn.SiLU(),
        )

        self.f2 = nn.Sequential(
            GhostConv3D(in_channels, out_channels, downscale=False, ratio=2),
            nn.GroupNorm(4, out_channels),
            nn.SiLU(),
        )

        self.f3 = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=1, padding=0),
            nn.GroupNorm(4, out_channels),
            nn.SiLU(),
        )

        self.se = SEBlock(out_channels)

    def create_spb(self, anchors, volume_shape, device="cuda"):
        B, K, _ = anchors.shape
        D, H, W = volume_shape

        x_arr = torch.arange(D, device=device, dtype=anchors.dtype) / D
        y_arr = torch.arange(H, device=device, dtype=anchors.dtype) / H
        z_arr = torch.arange(W, device=device, dtype=anchors.dtype) / W

        x = anchors[:, :, 0]
        y = anchors[:, :, 1]
        z = anchors[:, :, 2]

        x_diff = x_arr.view(1, 1, D, 1, 1) - x.view(B, K, 1, 1, 1)
        y_diff = y_arr.view(1, 1, 1, H, 1) - y.view(B, K, 1, 1, 1)
        z_diff = z_arr.view(1, 1, 1, 1, W) - z.view(B, K, 1, 1, 1)

        dists = x_diff + y_diff + z_diff
        return self.spb_conv(dists)

    def forward(self, input, skip, anchors):
        u = self.upsample_conv(input)
        spb = self.create_spb(anchors, u.shape[2:], device=u.device)

        d_in = self.fuse_conv(torch.cat([u + spb, skip], dim=1))
        pi = self.blending(d_in)
        f1_out = self.f1(d_in)
        f2_out = self.f2(d_in)
        f3_out = self.f3(d_in)
        f_out = (
            pi[:, 0:1, :, :, :] * f1_out
            + pi[:, 1:2, :, :, :] * f2_out
            + pi[:, 2:3, :, :, :] * f3_out
        )
        out = self.se(f_out)
        return out