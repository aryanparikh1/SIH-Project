"""
SwinIR4x — Swin Transformer for 4x Image Super-Resolution.
Matches the Kaggle training notebook architecture exactly.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


WINDOW_SIZE = 8  # Must match training


def window_partition(x, window_size):
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous()
    return windows.view(-1, window_size, window_size, C)


def window_reverse(windows, window_size, H, W):
    num_windows_per_image = (H // window_size) * (W // window_size)
    B = windows.shape[0] // num_windows_per_image
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
    return x.view(B, H, W, -1)


class WindowAttention(nn.Module):
    def __init__(self, dim, window_size, num_heads, dropout=0.0):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3)
        self.attn_drop = nn.Dropout(dropout)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(dropout)

        relative_size = 2 * window_size - 1
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros(relative_size * relative_size, num_heads)
        )

        coords_h = torch.arange(window_size)
        coords_w = torch.arange(window_size)
        coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing="ij"))
        coords_flatten = torch.flatten(coords, 1)

        relative_coords = (coords_flatten[:, :, None] - coords_flatten[:, None, :]).permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += window_size - 1
        relative_coords[:, :, 1] += window_size - 1
        relative_coords[:, :, 0] *= relative_size
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index, persistent=False)

        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

    def forward(self, x, mask=None):
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)

        relative_bias = self.relative_position_bias_table[
            self.relative_position_index.reshape(-1)
        ].reshape(N, N, self.num_heads).permute(2, 0, 1)
        attn = attn + relative_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N)
            attn = attn + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


def build_shift_mask(H, W, window_size, shift_size, device):
    if shift_size == 0:
        return None
    img_mask = torch.zeros((1, H, W, 1), device=device)
    h_slices = (slice(0, -window_size), slice(-window_size, -shift_size), slice(-shift_size, None))
    w_slices = (slice(0, -window_size), slice(-window_size, -shift_size), slice(-shift_size, None))
    cnt = 0
    for h in h_slices:
        for w in w_slices:
            img_mask[:, h, w, :] = cnt
            cnt += 1
    mask_windows = window_partition(img_mask, window_size).view(-1, window_size * window_size)
    attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
    attn_mask = attn_mask.masked_fill(attn_mask != 0, -100.0)
    attn_mask = attn_mask.masked_fill(attn_mask == 0, 0.0)
    return attn_mask


class SwinTransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, window_size=8, shift_size=0, dropout=0.1):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size

        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, window_size, num_heads, dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        B, H, W, C = x.shape
        shortcut = x
        x = self.norm1(x)

        pad_b = (self.window_size - H % self.window_size) % self.window_size
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        if pad_b > 0 or pad_r > 0:
            x = F.pad(x, (0, 0, 0, pad_r, 0, pad_b))
        Hp, Wp = x.shape[1], x.shape[2]

        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        x_windows = window_partition(shifted_x, self.window_size)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)

        attn_mask = build_shift_mask(Hp, Wp, self.window_size, self.shift_size, x.device)
        attn_windows = self.attn(x_windows, mask=attn_mask)
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)

        shifted_x = window_reverse(attn_windows, self.window_size, Hp, Wp)

        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x

        if pad_b > 0 or pad_r > 0:
            x = x[:, :H, :W, :].contiguous()

        x = shortcut + x
        x = x + self.mlp(self.norm2(x))
        return x


class SwinIR4x(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, embed_dim=96, depth=6, heads=6, scale=4, dropout=0.1):
        super().__init__()
        self.scale = scale

        self.head = nn.Conv2d(in_channels, embed_dim, kernel_size=3, padding=1)

        self.blocks = nn.ModuleList([
            SwinTransformerBlock(
                dim=embed_dim, num_heads=heads, window_size=WINDOW_SIZE,
                shift_size=(0 if i % 2 == 0 else WINDOW_SIZE // 2),
                dropout=dropout
            )
            for i in range(depth)
        ])

        self.norm = nn.LayerNorm(embed_dim)
        self.body = nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1)

        self.upsample = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim * (scale ** 2), kernel_size=3, padding=1),
            nn.PixelShuffle(scale),
            nn.GELU(),
            nn.Conv2d(embed_dim, out_channels, kernel_size=3, padding=1)
        )

    def forward(self, x):
        residual = F.interpolate(x, scale_factor=self.scale, mode="bicubic", align_corners=False)

        x = self.head(x)
        x_in = x.permute(0, 2, 3, 1)  # B, H, W, C

        for blk in self.blocks:
            x_in = blk(x_in)

        x_in = self.norm(x_in)
        x_body = self.body(x_in.permute(0, 3, 1, 2))  # B, C, H, W

        out = self.upsample(x + x_body)
        return torch.clamp(out + residual, 0.0, 1.0)


# Backward-compatible alias
SimpleTransformerSR = SwinIR4x
