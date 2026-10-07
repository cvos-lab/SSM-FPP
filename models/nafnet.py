"""NAFNet baseline (Chen et al., ECCV 2022), reimplemented in PyTorch following
the paper and the official repository (github.com/megvii-research/NAFNet, MIT).

Configuration: the published SIDD width-32 setting (width 32, encoder blocks
[2, 2, 4, 8], 12 middle blocks, decoder blocks [2, 2, 2, 2]); 29.16 M parameters
with a 6-channel input.

Adapted for 6-channel input -> 1-channel depth: the global input-output
residual of the original is removed, and the output is a 3x3 convolution
followed by a sigmoid (normalised depth in [0, 1]).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


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


class SimpleGate(nn.Module):
    """Element-wise multiplication of two halves along the channel dim."""
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class SimplifiedChannelAttention(nn.Module):
    """NAFNet's simplified channel attention: pool -> 1x1 conv -> multiply."""
    def __init__(self, channels):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv2d(channels, channels, kernel_size=1, bias=True)

    def forward(self, x):
        return x * self.conv(self.pool(x))


class NAFBlock(nn.Module):
    """The core NAFNet block: no nonlinear activation, uses SimpleGate + SCA."""
    def __init__(self, channels, dw_expand=2, ffn_expand=2, drop_prob=0.0):
        super().__init__()
        dw_ch = channels * dw_expand
        ffn_ch = channels * ffn_expand

        # --- Spatial branch ---
        self.norm1 = LayerNorm2d(channels)
        self.conv1 = nn.Conv2d(channels, dw_ch, kernel_size=1, bias=True)
        self.conv2 = nn.Conv2d(dw_ch, dw_ch, kernel_size=3, padding=1,
                               groups=dw_ch, bias=True)  # depthwise
        self.sg1 = SimpleGate()
        self.sca = SimplifiedChannelAttention(dw_ch // 2)
        self.conv3 = nn.Conv2d(dw_ch // 2, channels, kernel_size=1, bias=True)

        # --- Channel/FFN branch ---
        self.norm2 = LayerNorm2d(channels)
        self.conv4 = nn.Conv2d(channels, ffn_ch, kernel_size=1, bias=True)
        self.sg2 = SimpleGate()
        self.conv5 = nn.Conv2d(ffn_ch // 2, channels, kernel_size=1, bias=True)

        self.beta = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.gamma = nn.Parameter(torch.zeros(1, channels, 1, 1))

        self.drop = nn.Dropout2d(drop_prob) if drop_prob > 0 else nn.Identity()

    def forward(self, x):
        # Spatial branch
        y = self.norm1(x)
        y = self.conv1(y)
        y = self.conv2(y)
        y = self.sg1(y)
        y = self.sca(y)
        y = self.conv3(y)
        y = self.drop(y)
        x = x + y * self.beta

        # Channel branch
        y = self.norm2(x)
        y = self.conv4(y)
        y = self.sg2(y)
        y = self.conv5(y)
        y = self.drop(y)
        x = x + y * self.gamma
        return x


class NAFNet(nn.Module):
    def __init__(self,
                 in_channels=6,
                 out_channels=1,
                 width=32,
                 enc_blocks=(2, 2, 4, 8),
                 middle_blocks=12,
                 dec_blocks=(2, 2, 2, 2),
                 drop_prob=0.0):
        super().__init__()

        self.intro = nn.Conv2d(in_channels, width, kernel_size=3, padding=1, bias=True)

        # Encoder
        self.encoders = nn.ModuleList()
        self.downs = nn.ModuleList()
        ch = width
        for n in enc_blocks:
            self.encoders.append(nn.Sequential(*[NAFBlock(ch, drop_prob=drop_prob) for _ in range(n)]))
            self.downs.append(nn.Conv2d(ch, ch * 2, kernel_size=2, stride=2))
            ch *= 2

        # Middle
        self.middle = nn.Sequential(*[NAFBlock(ch, drop_prob=drop_prob) for _ in range(middle_blocks)])

        # Decoder
        self.ups = nn.ModuleList()
        self.decoders = nn.ModuleList()
        for n in dec_blocks:
            self.ups.append(nn.Sequential(
                nn.Conv2d(ch, ch * 2, kernel_size=1, bias=False),
                nn.PixelShuffle(2)
            ))
            ch //= 2
            self.decoders.append(nn.Sequential(*[NAFBlock(ch, drop_prob=drop_prob) for _ in range(n)]))

        self.final_conv = nn.Conv2d(width, out_channels, kernel_size=3, padding=1, bias=True)

        self.padder_size = 2 ** len(enc_blocks)

    def _check_image_size(self, x):
        _, _, h, w = x.shape
        pad_h = (self.padder_size - h % self.padder_size) % self.padder_size
        pad_w = (self.padder_size - w % self.padder_size) % self.padder_size
        x = F.pad(x, (0, pad_w, 0, pad_h))
        return x, (h, w)

    def forward(self, x):
        x, (H, W) = self._check_image_size(x)

        x = self.intro(x)
        skips = []
        for enc, down in zip(self.encoders, self.downs):
            x = enc(x)
            skips.append(x)
            x = down(x)

        x = self.middle(x)

        for up, dec, skip in zip(self.ups, self.decoders, reversed(skips)):
            x = up(x)
            x = x + skip
            x = dec(x)

        x = self.final_conv(x)
        x = x[:, :, :H, :W]              # crop pad
        return torch.sigmoid(x)