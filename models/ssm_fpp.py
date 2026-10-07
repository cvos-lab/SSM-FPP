"""SSM-FPP: multiscale encoder-decoder with a selective state-space context mixer.

Block template (Fig. 2b):
    F' = F + G(M(LN(F)))                  M: context mixer, G: channel gate
    F_out = F' + W2 GELU(W1 LN(F'))       1x1 convolutions, C -> 2C -> C

The context mixer M is selected with ``mixer``:
    "ssm"   selective state-space module (Mamba), the SSM-FPP default
    "conv"  expanded depthwise convolution: 1x1 C->3C, depthwise 3x3, GELU, 1x1 3C->C
    "mdta"  multi-Dconv head transposed attention (4 heads)
    "none"  identity
Everything outside M is identical across the four variants.

Requires mamba-ssm (CUDA).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from mamba_ssm import Mamba
except ImportError:                       # only needed for mixer="ssm"
    Mamba = None


class LayerNorm2d(nn.Module):
    """Layer normalisation over the channel dimension at each pixel."""
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


class ConvMixer2D(nn.Module):
    """Expanded depthwise convolution: 1x1 C->eC, depthwise 3x3, GELU, 1x1 eC->C."""
    def __init__(self, dim, expand=3):
        super().__init__()
        h = dim * expand
        self.net = nn.Sequential(
            nn.Conv2d(dim, h, 1),
            nn.Conv2d(h, h, 3, padding=1, groups=h),
            nn.GELU(),
            nn.Conv2d(h, dim, 1),
        )

    def forward(self, x):
        return self.net(x)


class MDTAMixer(nn.Module):
    """Multi-Dconv head transposed attention (Zamir et al., CVPR 2022):
    attention computed across channels."""
    def __init__(self, dim, num_heads=4, bias=True):
        super().__init__()
        while dim % num_heads != 0 and num_heads > 1:
            num_heads -= 1
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.qkv = nn.Conv2d(dim, dim * 3, 1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, 3, padding=1,
                                    groups=dim * 3, bias=bias)
        self.proj = nn.Conv2d(dim, dim, 1, bias=bias)

    def forward(self, x):
        B, C, H, W = x.shape
        q, k, v = self.qkv_dwconv(self.qkv(x)).chunk(3, dim=1)
        h = self.num_heads
        q = F.normalize(q.reshape(B, h, C // h, H * W), dim=-1)
        k = F.normalize(k.reshape(B, h, C // h, H * W), dim=-1)
        v = v.reshape(B, h, C // h, H * W)
        attn = ((q @ k.transpose(-2, -1)) * self.temperature).softmax(dim=-1)
        return self.proj((attn @ v).reshape(B, C, H, W))


class SSMFPPBlock(nn.Module):
    """LN -> context mixer -> channel gate -> residual; LN -> FFN -> residual."""
    def __init__(self, dim, d_state=16, d_conv=4, expand=2, mixer="ssm"):
        super().__init__()
        self.norm = LayerNorm2d(dim)
        self.mixer_kind = mixer
        self.ssm = None
        if mixer == "ssm":
            if Mamba is None:
                raise ImportError("mixer='ssm' requires mamba-ssm (pip install mamba-ssm)")
            self.ssm = Mamba(d_model=dim, d_state=d_state, d_conv=d_conv, expand=expand)
        elif mixer == "conv":
            self.mix_conv = ConvMixer2D(dim, expand=3)
        elif mixer == "mdta":
            self.mix_attn = MDTAMixer(dim)
        elif mixer != "none":
            raise ValueError(f"unknown mixer {mixer!r}")

        # channel gate: global average pool -> 1x1 conv -> sigmoid
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, dim, kernel_size=1),
            nn.Sigmoid(),
        )
        self.ffn_norm = LayerNorm2d(dim)
        self.ffn = nn.Sequential(
            nn.Conv2d(dim, dim * 2, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(dim * 2, dim, kernel_size=1),
        )

    def forward(self, x):
        # Kept in the form used for the reported measurements: the explicit
        # intermediates fix the peak-memory figure given by
        # tools/measure_efficiency.py (outputs are identical either way).
        B, C, H, W = x.shape
        residual = x
        y = self.norm(x)
        if self.mixer_kind == "ssm":
            # raster-order flattening: (B, C, H, W) -> (B, HW, C) -> Mamba -> back
            tokens = y.flatten(2).transpose(1, 2)
            tokens = self.ssm(tokens)
            y = tokens.transpose(1, 2).reshape(B, C, H, W)
        elif self.mixer_kind == "conv":
            y = self.mix_conv(y)
        elif self.mixer_kind == "mdta":
            y = self.mix_attn(y)
        y = y * self.gate(y)
        x = residual + y
        x = x + self.ffn(self.ffn_norm(x))
        return x


class Downsample(nn.Module):
    """3x3 conv C -> C/2, then pixel-unshuffle: half resolution, 2C channels."""
    def __init__(self, ch):
        super().__init__()
        self.body = nn.Sequential(nn.Conv2d(ch, ch // 2, 3, padding=1, bias=False),
                                  nn.PixelUnshuffle(2))

    def forward(self, x):
        return self.body(x)


class Upsample(nn.Module):
    """3x3 conv C -> 2C, then pixel-shuffle: double resolution, C/2 channels."""
    def __init__(self, ch):
        super().__init__()
        self.body = nn.Sequential(nn.Conv2d(ch, ch * 2, 3, padding=1, bias=False),
                                  nn.PixelShuffle(2))

    def forward(self, x):
        return self.body(x)


class SSMFPP(nn.Module):
    """Four-level encoder-decoder. Blocks per level ``num_blocks`` =
    (encoder 1, encoder 2, encoder 3, bottleneck); the decoder mirrors the
    encoder. Output: normalised depth in [0, 1] (3x3 conv + sigmoid)."""
    def __init__(self, in_channels=6, out_channels=1, dim=48, num_blocks=(2, 4, 4, 6),
                 d_state=16, d_conv=4, expand=2, mixer="ssm"):
        super().__init__()

        def stage(ch, n):
            return nn.Sequential(*[SSMFPPBlock(ch, d_state, d_conv, expand, mixer)
                                   for _ in range(n)])

        self.patch_embed = nn.Conv2d(in_channels, dim, kernel_size=3, padding=1)

        self.enc1 = stage(dim, num_blocks[0])
        self.down1 = Downsample(dim)
        self.enc2 = stage(dim * 2, num_blocks[1])
        self.down2 = Downsample(dim * 2)
        self.enc3 = stage(dim * 4, num_blocks[2])
        self.down3 = Downsample(dim * 4)
        self.bottleneck = stage(dim * 8, num_blocks[3])

        self.up3 = Upsample(dim * 8)
        self.reduce3 = nn.Conv2d(dim * 8, dim * 4, kernel_size=1)
        self.dec3 = stage(dim * 4, num_blocks[2])
        self.up2 = Upsample(dim * 4)
        self.reduce2 = nn.Conv2d(dim * 4, dim * 2, kernel_size=1)
        self.dec2 = stage(dim * 2, num_blocks[1])
        self.up1 = Upsample(dim * 2)
        self.reduce1 = nn.Conv2d(dim * 2, dim, kernel_size=1)
        self.dec1 = stage(dim, num_blocks[0])

        self.final_conv = nn.Conv2d(dim, out_channels, kernel_size=3, padding=1)
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
        b = self.bottleneck(self.down3(e3))

        d3 = self.up3(b)
        d3 = self.dec3(self.reduce3(torch.cat([d3, e3], dim=1)))

        d2 = self.up2(d3)
        d2 = self.dec2(self.reduce2(torch.cat([d2, e2], dim=1)))

        d1 = self.up1(d2)
        d1 = self.dec1(self.reduce1(torch.cat([d1, e1], dim=1)))

        out = self.final_conv(d1)
        out = out[:, :, :H, :W]
        return torch.sigmoid(out)
