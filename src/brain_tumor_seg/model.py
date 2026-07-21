from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class DoubleConv(nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dropout: float = 0.0,
        batch_norm: bool = True,
    ) -> None:
        layers: list[nn.Module] = []
        for input_channels in (in_channels, out_channels):
            layers.append(
                nn.Conv2d(
                    input_channels,
                    out_channels,
                    kernel_size=3,
                    padding=1,
                    bias=not batch_norm,
                )
            )
            if batch_norm:
                layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU(inplace=True))
        if dropout > 0:
            layers.append(nn.Dropout2d(dropout))
        super().__init__(*layers)


class Down(nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dropout: float = 0.0,
        batch_norm: bool = True,
    ) -> None:
        super().__init__(
            nn.MaxPool2d(2), DoubleConv(in_channels, out_channels, dropout, batch_norm)
        )


class Up(nn.Module):
    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        dropout: float = 0.0,
        batch_norm: bool = True,
    ) -> None:
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = DoubleConv(
            out_channels + skip_channels, out_channels, dropout, batch_norm
        )

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
        batch_norm: bool = True,
    ) -> None:
        super().__init__()
        channels = [base_channels * (2**index) for index in range(5)]
        self.inc = DoubleConv(in_channels, channels[0], 0.0, batch_norm)
        self.down1 = Down(channels[0], channels[1], 0.0, batch_norm)
        self.down2 = Down(channels[1], channels[2], dropout, batch_norm)
        self.down3 = Down(channels[2], channels[3], dropout, batch_norm)
        self.down4 = Down(channels[3], channels[4], dropout, batch_norm)
        self.up1 = Up(channels[4], channels[3], channels[3], dropout, batch_norm)
        self.up2 = Up(channels[3], channels[2], channels[2], dropout, batch_norm)
        self.up3 = Up(channels[2], channels[1], channels[1], 0.0, batch_norm)
        self.up4 = Up(channels[1], channels[0], channels[0], 0.0, batch_norm)
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


class ResNet34UNet(nn.Module):
    """U-Net decoder using torchvision's ResNet-34 feature pyramid as encoder."""

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        dropout: float = 0.0,
        batch_norm: bool = True,
        pretrained: bool = True,
    ) -> None:
        super().__init__()
        try:
            from torchvision.models import ResNet34_Weights, resnet34
        except ImportError as error:
            raise ImportError(
                "The ResNet-34 encoder requires torchvision. Install the project dependencies "
                "with: python -m pip install -e ."
            ) from error

        weights = ResNet34_Weights.DEFAULT if pretrained else None
        backbone = resnet34(weights=weights)
        if in_channels != 3:
            if in_channels != 1:
                raise ValueError("ResNet-34 encoder supports only 1-channel or 3-channel input")
            original_conv = backbone.conv1
            conv1 = nn.Conv2d(
                1,
                original_conv.out_channels,
                kernel_size=original_conv.kernel_size,
                stride=original_conv.stride,
                padding=original_conv.padding,
                bias=False,
            )
            with torch.no_grad():
                conv1.weight.copy_(original_conv.weight.mean(dim=1, keepdim=True))
            backbone.conv1 = conv1

        self.stem = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu)
        self.pool = backbone.maxpool
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4

        self.up1 = Up(512, 256, 256, dropout, batch_norm)
        self.up2 = Up(256, 128, 128, dropout, batch_norm)
        self.up3 = Up(128, 64, 64, dropout, batch_norm)
        self.up4 = Up(64, 64, 32, 0.0, batch_norm)
        self.up5 = nn.Sequential(
            nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2),
            DoubleConv(16, 16, 0.0, batch_norm),
        )
        self.outc = nn.Conv2d(16, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        stem = self.stem(x)
        x = self.pool(stem)
        skip1 = self.layer1(x)
        skip2 = self.layer2(skip1)
        skip3 = self.layer3(skip2)
        bottleneck = self.layer4(skip3)
        x = self.up1(bottleneck, skip3)
        x = self.up2(x, skip2)
        x = self.up3(x, skip1)
        x = self.up4(x, stem)
        return self.outc(self.up5(x))


def build_model(config: dict[str, object]) -> nn.Module:
    name = str(config.get("name", "unet")).lower()
    encoder = str(config.get("encoder", "double_conv")).lower()
    if name in {"resnet34_unet", "resnet34unet"}:
        name = "unet"
        encoder = "resnet34"
    if name == "unet" and encoder == "resnet34":
        return ResNet34UNet(
            in_channels=int(config["in_channels"]),
            out_channels=int(config["out_channels"]),
            dropout=float(config.get("dropout", 0.0)),
            batch_norm=bool(config.get("batch_norm", True)),
            pretrained=bool(config.get("pretrained", True)),
        )
    if name != "unet" or encoder != "double_conv":
        raise ValueError(f"Unsupported model: {name}")
    return UNet(
        in_channels=int(config["in_channels"]),
        out_channels=int(config["out_channels"]),
        base_channels=int(config["base_channels"]),
        dropout=float(config.get("dropout", 0.0)),
        batch_norm=bool(config.get("batch_norm", True)),
    )
