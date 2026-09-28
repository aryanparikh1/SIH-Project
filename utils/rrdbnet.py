"""
RRDBNet architecture matching the Kaggle training notebook exactly.

Key details from training code:
- residual_scale = 0.2 in both ResidualDenseBlock and RRDB
- Single Dropout2d(p=0.1) placed AFTER the trunk skip connection
- output_activation = "sigmoid" applied at the end of forward()
- Weight initialization: kaiming_normal_ with a=0.2, scaled by 0.1
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualDenseBlock(nn.Module):
    def __init__(self, num_feat=64, growth_channels=32, residual_scale=0.2):
        super().__init__()
        gc = growth_channels
        self.residual_scale = residual_scale
        self.conv1 = nn.Conv2d(num_feat, gc, 3, 1, 1)
        self.conv2 = nn.Conv2d(num_feat + gc, gc, 3, 1, 1)
        self.conv3 = nn.Conv2d(num_feat + 2 * gc, gc, 3, 1, 1)
        self.conv4 = nn.Conv2d(num_feat + 3 * gc, gc, 3, 1, 1)
        self.conv5 = nn.Conv2d(num_feat + 4 * gc, num_feat, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        f1 = self.lrelu(self.conv1(x))
        f2 = self.lrelu(self.conv2(torch.cat([x, f1], dim=1)))
        f3 = self.lrelu(self.conv3(torch.cat([x, f1, f2], dim=1)))
        f4 = self.lrelu(self.conv4(torch.cat([x, f1, f2, f3], dim=1)))
        f5 = self.conv5(torch.cat([x, f1, f2, f3, f4], dim=1))
        return x + self.residual_scale * f5


class RRDB(nn.Module):
    def __init__(self, num_feat=64, growth_channels=32, residual_scale=0.2):
        super().__init__()
        self.residual_scale = residual_scale
        self.rdb1 = ResidualDenseBlock(num_feat, growth_channels, residual_scale)
        self.rdb2 = ResidualDenseBlock(num_feat, growth_channels, residual_scale)
        self.rdb3 = ResidualDenseBlock(num_feat, growth_channels, residual_scale)

    def forward(self, x):
        out = self.rdb1(x)
        out = self.rdb2(out)
        out = self.rdb3(out)
        return x + self.residual_scale * out


class RRDBNet(nn.Module):
    def __init__(
        self,
        in_channels=4,
        out_channels=4,
        num_features=64,
        growth_channels=32,
        num_rrdb_blocks=6,
        residual_scale=0.2,
        output_activation="sigmoid",
        dropout_p=0.1,
        # Legacy kwargs for backward compat with old loader calls
        in_nc=None, out_nc=None, nf=None, nb=None, gc=None,
    ):
        super().__init__()

        # Support old-style kwargs
        if in_nc is not None:
            in_channels = in_nc
        if out_nc is not None:
            out_channels = out_nc
        if nf is not None:
            num_features = nf
        if nb is not None:
            num_rrdb_blocks = nb
        if gc is not None:
            growth_channels = gc

        self.output_activation = output_activation
        self.dropout_p = dropout_p

        self.conv_first = nn.Conv2d(in_channels, num_features, 3, 1, 1)

        self.trunk = nn.Sequential(*[
            RRDB(num_features, growth_channels, residual_scale)
            for _ in range(num_rrdb_blocks)
        ])

        self.conv_trunk = nn.Conv2d(num_features, num_features, 3, 1, 1)

        self.dropout = nn.Dropout2d(p=dropout_p)

        self.upconv1 = nn.Conv2d(num_features, num_features, 3, 1, 1)
        self.upconv2 = nn.Conv2d(num_features, num_features, 3, 1, 1)
        self.conv_hr = nn.Conv2d(num_features, num_features, 3, 1, 1)
        self.conv_last = nn.Conv2d(num_features, out_channels, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        feat = self.conv_first(x)
        trunk = self.conv_trunk(self.trunk(feat))
        feat = feat + trunk
        feat = self.dropout(feat)

        feat = F.interpolate(feat, scale_factor=2, mode="nearest")
        feat = self.lrelu(self.upconv1(feat))
        feat = F.interpolate(feat, scale_factor=2, mode="nearest")
        feat = self.lrelu(self.upconv2(feat))
        feat = self.lrelu(self.conv_hr(feat))
        out = self.conv_last(feat)

        if self.output_activation == "sigmoid":
            out = torch.sigmoid(out)

        return out
