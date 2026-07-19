from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class DoubleConv(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.0) -> None:
        layers: list[nn.Module] = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]
        if dropout > 0:
            layers.append(nn.Dropout2d(dropout))
        super().__init__(*layers)


class Down(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, dropout: float) -> None:
        super().__init__(nn.MaxPool2d(2), DoubleConv(in_channels, out_channels, dropout))


class Up(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int, dropout: float) -> None:
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = DoubleConv(out_channels + skip_channels, out_channels, dropout)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        diff_y = skip.size(2) - x.size(2)
        diff_x = skip.size(3) - x.size(3)
        if diff_x or diff_y:
            x = F.pad(
                x,
                [diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2],
            )
        return self.conv(torch.cat([skip, x], dim=1))


class UNet(nn.Module):
    """Four-level 2D U-Net returning raw binary segmentation logits."""

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        base_channels: int = 32,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        channels = [base_channels * (2**index) for index in range(5)]
        self.inc = DoubleConv(in_channels, channels[0], 0.0)
        self.down1 = Down(channels[0], channels[1], 0.0)
        self.down2 = Down(channels[1], channels[2], dropout)
        self.down3 = Down(channels[2], channels[3], dropout)
        self.down4 = Down(channels[3], channels[4], dropout)
        self.up1 = Up(channels[4], channels[3], channels[3], dropout)
        self.up2 = Up(channels[3], channels[2], channels[2], dropout)
        self.up3 = Up(channels[2], channels[1], channels[1], 0.0)
        self.up4 = Up(channels[1], channels[0], channels[0], 0.0)
        self.outc = nn.Conv2d(channels[0], out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        return self.outc(x)


def build_model(config: dict[str, object]) -> nn.Module:
    name = str(config.get("name", "unet")).lower()
    if name != "unet":
        raise ValueError(f"Unsupported model: {name}")
    return UNet(
        in_channels=int(config["in_channels"]),
        out_channels=int(config["out_channels"]),
        base_channels=int(config["base_channels"]),
        dropout=float(config.get("dropout", 0.0)),
    )

