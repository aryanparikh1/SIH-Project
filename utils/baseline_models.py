import torch
import torch.nn as nn


class SRCNN(nn.Module):
    """
    SRCNN: 3-layer CNN for super-resolution.
    Architecture from weights: Conv(3->64, 9x9) -> ReLU -> Conv(64->32, 5x5) -> ReLU -> Conv(32->3, 5x5)
    """
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=9, padding=4),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 3, kernel_size=5, padding=2),
        )

    def forward(self, x):
        return self.features(x)


class ResBlock(nn.Module):
    """Residual block used in RUNet: Conv->ReLU->Conv with skip connection."""
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1),
        )

    def forward(self, x):
        return x + self.block(x)


class RUNet(nn.Module):
    """
    Residual U-Net for super-resolution.
    Encoder: enc1(3->64) -> enc2(64->128)
    Bottleneck: 2x ResBlock(128)
    Decoder: up(128->64) -> dec(128->64, from concat) -> ResBlock(64)
    Output: 1x1 conv(64->3)
    """
    def __init__(self):
        super().__init__()
        # Encoder
        self.enc1 = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            ResBlock(64),
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            ResBlock(128),
        )
        # Bottleneck
        self.bottleneck = nn.Sequential(
            ResBlock(128),
            ResBlock(128),
        )
        # Decoder
        self.up = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            ResBlock(64),
        )
        self.output = nn.Conv2d(64, 3, 1)

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(nn.functional.max_pool2d(e1, 2))
        # Bottleneck
        b = self.bottleneck(e2)
        # Decoder
        d = self.up(b)
        # Pad if needed to match e1 size
        diff_h = e1.size(2) - d.size(2)
        diff_w = e1.size(3) - d.size(3)
        d = nn.functional.pad(d, [diff_w // 2, diff_w - diff_w // 2,
                                   diff_h // 2, diff_h - diff_h // 2])
        d = torch.cat([d, e1], dim=1)
        d = self.dec(d)
        return self.output(d)


class EDSRResBlock(nn.Module):
    """EDSR residual block: Conv->ReLU->Conv with residual scaling."""
    def __init__(self, n_feats=64, res_scale=0.1):
        super().__init__()
        self.conv1 = nn.Conv2d(n_feats, n_feats, 3, padding=1)
        self.conv2 = nn.Conv2d(n_feats, n_feats, 3, padding=1)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.conv1(x)
        res = nn.functional.relu(res, inplace=True)
        res = self.conv2(res)
        res = res * self.res_scale
        return x + res


class EDSR(nn.Module):
    """
    EDSR: Enhanced Deep Residual Network for super-resolution.
    8 residual blocks, 64 features, 4x upscale via 2x PixelShuffle layers.
    """
    def __init__(self, n_resblocks=8, n_feats=64, scale=4):
        super().__init__()
        self.head = nn.Conv2d(3, n_feats, 3, padding=1)
        self.body = nn.Sequential(*[EDSRResBlock(n_feats) for _ in range(n_resblocks)])
        self.body_conv = nn.Conv2d(n_feats, n_feats, 3, padding=1)
        # 4x upscale = 2x PixelShuffle twice
        self.up1 = nn.Sequential(
            nn.Conv2d(n_feats, n_feats * 4, 3, padding=1),
            nn.PixelShuffle(2),
            nn.ReLU(inplace=True)
        )
        self.up2 = nn.Sequential(
            nn.Conv2d(n_feats, n_feats * 4, 3, padding=1),
            nn.PixelShuffle(2),
            nn.ReLU(inplace=True)
        )
        self.tail = nn.Conv2d(n_feats, 3, 3, padding=1)

    def forward(self, x):
        res = self.body_conv(self.body(self.head(x)))
        return self.tail(self.up2(self.up1(self.head(x) + res)))
