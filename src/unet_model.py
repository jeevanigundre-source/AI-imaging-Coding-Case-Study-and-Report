"""
unet_model.py -- a small U-Net for binary nucleus segmentation, plus the three
losses used in the ablation (BCE, soft Dice, BCE+Dice).

Deliberately small: with only 80 training images a full-width U-Net (64-1024
channels, ~31M parameters) would overfit almost immediately, so the base width
is 16 and the depth is 3 (~0.5M parameters).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as Fn


class DoubleConv(nn.Module):
    """(conv 3x3 -> BN -> ReLU) x 2 -- BatchNorm matters here because the batch is
    tiny and the foreground/background imbalance is large."""

    def __init__(self, cin: int, cout: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1, bias=False),
            nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
            nn.Conv2d(cout, cout, 3, padding=1, bias=False),
            nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class UNet(nn.Module):
    def __init__(self, in_ch: int = 1, out_ch: int = 1, base: int = 16, depth: int = 3):
        super().__init__()
        self.depth = depth
        chans = [base * 2 ** i for i in range(depth)]           # 16, 32, 64
        self.downs = nn.ModuleList()
        c = in_ch
        for ch in chans:
            self.downs.append(DoubleConv(c, ch))
            c = ch
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = DoubleConv(c, c * 2)                  # 128
        self.ups = nn.ModuleList()
        self.upconvs = nn.ModuleList()
        c = c * 2
        for ch in reversed(chans):
            self.upconvs.append(nn.ConvTranspose2d(c, ch, 2, stride=2))
            self.ups.append(DoubleConv(ch * 2, ch))             # skip concat
            c = ch
        self.head = nn.Conv2d(c, out_ch, 1)                     # logits

    def forward(self, x):
        skips = []
        for d in self.downs:
            x = d(x)
            skips.append(x)
            x = self.pool(x)
        x = self.bottleneck(x)
        for up, conv, skip in zip(self.upconvs, self.ups, reversed(skips)):
            x = up(x)
            if x.shape[-2:] != skip.shape[-2:]:
                x = Fn.interpolate(x, size=skip.shape[-2:], mode="nearest")
            x = conv(torch.cat([skip, x], dim=1))
        return self.head(x)

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


# ------------------------------------------------------------------- losses
def soft_dice_loss(logits: torch.Tensor, target: torch.Tensor,
                   eps: float = 1.0) -> torch.Tensor:
    """1 - soft Dice, computed per image then averaged (not over the whole batch),
    so a nearly-empty sparse image is not drowned out by a dense one."""
    p = torch.sigmoid(logits)
    dims = (1, 2, 3)
    inter = (p * target).sum(dims)
    denom = p.sum(dims) + target.sum(dims)
    return (1 - (2 * inter + eps) / (denom + eps)).mean()


def bce_loss(logits, target):
    return Fn.binary_cross_entropy_with_logits(logits, target)


def bce_dice_loss(logits, target, w: float = 0.5):
    return w * bce_loss(logits, target) + (1 - w) * soft_dice_loss(logits, target)


LOSSES = {"bce": bce_loss, "dice": soft_dice_loss, "bce_dice": bce_dice_loss}
