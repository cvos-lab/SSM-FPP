"""Restormer baseline (Zamir et al., CVPR 2022), reimplemented in PyTorch following
the paper and the official repository (github.com/swz30/Restormer, MIT).

Configuration: the published default (dim 48, blocks [4, 6, 6, 8], heads
[1, 2, 4, 8], GDFN expansion 2.66, 4 refinement blocks, bias off); 26.13 M
parameters with a 6-channel input. MDTA computes attention across channels.

Adapted for 6-channel input -> 1-channel depth: the global input-output
residual of the original is removed, and the output is a 3x3 convolution
followed by a sigmoid (normalised depth in [0, 1]).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class LayerNorm2d(nn.Module):
    def __init__(self, channels, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.eps = eps

    def forward(self, x):
        mu = x.mean(dim=1, keepdim=True)
        var = x.var(dim=1, keepdim=True, unbiased=False)
        x = (x - mu) / torch.sqrt(var + self.eps)
        return x * self.weight.view(1, -1, 1, 1) + self.bias.view(1, -1, 1, 1)


class MDTA(nn.Module):
    """Multi-Dconv Head Transposed Attention.
    Self-attention across channels via dot-product of K^T Q."""
    def __init__(self, dim, num_heads, bias=True):
        super().__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, padding=1,
                                    groups=dim * 3, bias=bias)
        self.proj = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        B, C, H, W = x.shape
        qkv = self.qkv_dwconv(self.qkv(x))                                  # (B, 3C, H, W)
        q, k, v = qkv.chunk(3, dim=1)
        q = rearrange(q, 'b (h c) x y -> b h c (x y)', h=self.num_heads)
        k = rearrange(k, 'b (h c) x y -> b h c (x y)', h=self.num_heads)
        v = rearrange(v, 'b (h c) x y -> b h c (x y)', h=self.num_heads)

        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature                 # (B, h, c, c)
        attn = attn.softmax(dim=-1)
        out = attn @ v                                                       # (B, h, c, HW)
        out = rearrange(out, 'b h c (x y) -> b (h c) x y', x=H, y=W)
        return self.proj(out)


class GDFN(nn.Module):
    """Gated-Dconv Feed-Forward Network."""
    def __init__(self, dim, ffn_expansion=2.66, bias=True):
        super().__init__()
        hidden = int(dim * ffn_expansion)
        self.proj_in = nn.Conv2d(dim, hidden * 2, kernel_size=1, bias=bias)
        self.dwconv = nn.Conv2d(hidden * 2, hidden * 2, kernel_size=3, padding=1,
                                groups=hidden * 2, bias=bias)
        self.proj_out = nn.Conv2d(hidden, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.proj_in(x)
        x = self.dwconv(x)
        x1, x2 = x.chunk(2, dim=1)
        x = F.gelu(x1) * x2
        return self.proj_out(x)


class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion=2.66, bias=True):
        super().__init__()
        self.norm1 = LayerNorm2d(dim)
        self.attn = MDTA(dim, num_heads, bias)
        self.norm2 = LayerNorm2d(dim)
        self.ffn = GDFN(dim, ffn_expansion, bias)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class Downsample(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(ch, ch // 2, kernel_size=3, padding=1, bias=False),
            nn.PixelUnshuffle(2)
        )

    def forward(self, x):
        return self.body(x)


class Upsample(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(ch, ch * 2, kernel_size=3, padding=1, bias=False),
            nn.PixelShuffle(2)
        )

    def forward(self, x):
        return self.body(x)


class Restormer(nn.Module):
    def __init__(self,
                 in_channels=6,
                 out_channels=1,
                 dim=48,
                 num_blocks=(4, 6, 6, 8),
                 num_heads=(1, 2, 4, 8),
                 ffn_expansion=2.66,
                 bias=False):
        super().__init__()

        self.patch_embed = nn.Conv2d(in_channels, dim, kernel_size=3, padding=1, bias=bias)

        # Encoder levels 1..4
        self.enc1 = nn.Sequential(*[TransformerBlock(dim, num_heads[0], ffn_expansion, bias)
                                     for _ in range(num_blocks[0])])
        self.down1 = Downsample(dim)

        self.enc2 = nn.Sequential(*[TransformerBlock(dim * 2, num_heads[1], ffn_expansion, bias)
                                     for _ in range(num_blocks[1])])
        self.down2 = Downsample(dim * 2)

        self.enc3 = nn.Sequential(*[TransformerBlock(dim * 4, num_heads[2], ffn_expansion, bias)
                                     for _ in range(num_blocks[2])])
        self.down3 = Downsample(dim * 4)

        # Bottleneck
        self.latent = nn.Sequential(*[TransformerBlock(dim * 8, num_heads[3], ffn_expansion, bias)
                                       for _ in range(num_blocks[3])])

        # Decoder
        self.up3 = Upsample(dim * 8)
        self.reduce3 = nn.Conv2d(dim * 8, dim * 4, kernel_size=1, bias=bias)
        self.dec3 = nn.Sequential(*[TransformerBlock(dim * 4, num_heads[2], ffn_expansion, bias)
                                     for _ in range(num_blocks[2])])

        self.up2 = Upsample(dim * 4)
        self.reduce2 = nn.Conv2d(dim * 4, dim * 2, kernel_size=1, bias=bias)
        self.dec2 = nn.Sequential(*[TransformerBlock(dim * 2, num_heads[1], ffn_expansion, bias)
                                     for _ in range(num_blocks[1])])

        self.up1 = Upsample(dim * 2)
        # No reduce here in original Restormer (concat keeps 2*dim)
        self.dec1 = nn.Sequential(*[TransformerBlock(dim * 2, num_heads[0], ffn_expansion, bias)
                                     for _ in range(num_blocks[0])])

        # Refinement + output
        self.refine = nn.Sequential(*[TransformerBlock(dim * 2, num_heads[0], ffn_expansion, bias)
                                       for _ in range(4)])
        self.final_conv = nn.Conv2d(dim * 2, out_channels, kernel_size=3, padding=1, bias=bias)

        self.padder_size = 8

    def _check_image_size(self, x):
        _, _, h, w = x.shape
        pad_h = (self.padder_size - h % self.padder_size) % self.padder_size
        pad_w = (self.padder_size - w % self.padder_size) % self.padder_size
        x = F.pad(x, (0, pad_w, 0, pad_h))
        return x, (h, w)

    def forward(self, x):
        x, (H, W) = self._check_image_size(x)

        x = self.patch_embed(x)
        e1 = self.enc1(x)
        e2 = self.enc2(self.down1(e1))
        e3 = self.enc3(self.down2(e2))

        b = self.latent(self.down3(e3))

        d3 = self.up3(b)
        d3 = self.dec3(self.reduce3(torch.cat([d3, e3], dim=1)))

        d2 = self.up2(d3)
        d2 = self.dec2(self.reduce2(torch.cat([d2, e2], dim=1)))

        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))

        out = self.refine(d1)
        out = self.final_conv(out)
        out = out[:, :, :H, :W]
        return torch.sigmoid(out)