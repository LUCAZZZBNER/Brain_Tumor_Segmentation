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


class AttentionGate(nn.Module):
    """Filter a skip feature map using a decoder-derived gating signal."""

    def __init__(
        self,
        gate_channels: int,
        skip_channels: int,
        intermediate_channels: int,
        batch_norm: bool = True,
    ) -> None:
        super().__init__()

        def projection(in_channels: int) -> nn.Sequential:
            layers: list[nn.Module] = [
                nn.Conv2d(in_channels, intermediate_channels, kernel_size=1, bias=not batch_norm)
            ]
            if batch_norm:
                layers.append(nn.BatchNorm2d(intermediate_channels))
            return nn.Sequential(*layers)

        self.gate_projection = projection(gate_channels)
        self.skip_projection = projection(skip_channels)
        self.activation = nn.ReLU(inplace=True)
        self.attention = nn.Sequential(
            nn.Conv2d(intermediate_channels, 1, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, gate: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        if gate.shape[-2:] != skip.shape[-2:]:
            gate = F.interpolate(gate, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        coefficients = self.attention(
            self.activation(self.gate_projection(gate) + self.skip_projection(skip))
        )
        return skip * coefficients


class AttentionUp(nn.Module):
    """Upsample decoder features and attention-filter the matching skip connection."""

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
        self.gate = AttentionGate(
            gate_channels=out_channels,
            skip_channels=skip_channels,
            intermediate_channels=max(skip_channels // 2, 1),
            batch_norm=batch_norm,
        )
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
        filtered_skip = self.gate(x, skip)
        return self.conv(torch.cat([filtered_skip, x], dim=1))


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


class ASPP(nn.Module):
    """Atrous spatial pyramid pooling for compact bottleneck feature maps."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        branch_channels: int,
        dilation_rates: tuple[int, ...] = (1, 2, 4, 8),
        dropout: float = 0.1,
        batch_norm: bool = True,
    ) -> None:
        super().__init__()
        if in_channels <= 0 or out_channels <= 0 or branch_channels <= 0:
            raise ValueError("ASPP channel counts must be positive")
        if not dilation_rates or any(rate <= 0 for rate in dilation_rates):
            raise ValueError("ASPP dilation rates must be non-empty positive integers")

        self.dilation_rates = dilation_rates
        self.branches = nn.ModuleList(
            [
                self._conv_branch(
                    in_channels,
                    branch_channels,
                    kernel_size=1,
                    dilation=1,
                    batch_norm=batch_norm,
                ),
                *[
                    self._conv_branch(
                        in_channels,
                        branch_channels,
                        kernel_size=3,
                        dilation=rate,
                        batch_norm=batch_norm,
                    )
                    for rate in dilation_rates
                ],
            ]
        )
        # BatchNorm is deliberately omitted after global 1x1 pooling so training also works
        # when the final batch contains a single image.
        self.global_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, branch_channels, kernel_size=1),
            nn.ReLU(inplace=True),
        )

        merged_channels = branch_channels * (len(self.branches) + 1)
        projection: list[nn.Module] = [
            nn.Conv2d(merged_channels, out_channels, kernel_size=1, bias=not batch_norm)
        ]
        if batch_norm:
            projection.append(nn.BatchNorm2d(out_channels))
        projection.append(nn.ReLU(inplace=True))
        if dropout > 0:
            projection.append(nn.Dropout2d(dropout))
        self.project = nn.Sequential(*projection)

    @staticmethod
    def _conv_branch(
        in_channels: int,
        out_channels: int,
        *,
        kernel_size: int,
        dilation: int,
        batch_norm: bool,
    ) -> nn.Sequential:
        layers: list[nn.Module] = [
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                padding=dilation if kernel_size == 3 else 0,
                dilation=dilation,
                bias=not batch_norm,
            )
        ]
        if batch_norm:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU(inplace=True))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        spatial_size = x.shape[-2:]
        features = [branch(x) for branch in self.branches]
        pooled = F.interpolate(
            self.global_pool(x), size=spatial_size, mode="bilinear", align_corners=False
        )
        return self.project(torch.cat([*features, pooled], dim=1))


class ASPPUNet(nn.Module):
    """Original U-Net with multi-scale ASPP context at its bottleneck."""

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        base_channels: int = 32,
        dropout: float = 0.1,
        batch_norm: bool = True,
        aspp_branch_channels: int | None = None,
        aspp_dilation_rates: tuple[int, ...] = (1, 2, 4, 8),
        aspp_dropout: float | None = None,
    ) -> None:
        super().__init__()
        channels = [base_channels * (2**index) for index in range(5)]
        branch_channels = (
            max(channels[4] // 4, 1)
            if aspp_branch_channels is None
            else aspp_branch_channels
        )
        self.inc = DoubleConv(in_channels, channels[0], 0.0, batch_norm)
        self.down1 = Down(channels[0], channels[1], 0.0, batch_norm)
        self.down2 = Down(channels[1], channels[2], dropout, batch_norm)
        self.down3 = Down(channels[2], channels[3], dropout, batch_norm)
        self.down4 = Down(channels[3], channels[4], dropout, batch_norm)
        self.aspp = ASPP(
            channels[4],
            channels[4],
            branch_channels,
            dilation_rates=aspp_dilation_rates,
            dropout=dropout if aspp_dropout is None else aspp_dropout,
            batch_norm=batch_norm,
        )
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
        x5 = self.aspp(self.down4(x4))
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        return self.outc(x)


class AttentionUNet(nn.Module):
    """Original U-Net encoder/decoder with Attention Gates on all skip connections."""

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
        self.up1 = AttentionUp(channels[4], channels[3], channels[3], dropout, batch_norm)
        self.up2 = AttentionUp(channels[3], channels[2], channels[2], dropout, batch_norm)
        self.up3 = AttentionUp(channels[2], channels[1], channels[1], 0.0, batch_norm)
        self.up4 = AttentionUp(channels[1], channels[0], channels[0], 0.0, batch_norm)
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


class UNetPlusPlus(nn.Module):
    """U-Net++ with nested dense skip pathways and a single final segmentation head."""

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        base_channels: int = 32,
        dropout: float = 0.1,
        batch_norm: bool = True,
    ) -> None:
        super().__init__()
        self.channels = [base_channels * (2**index) for index in range(5)]
        self.inc = DoubleConv(in_channels, self.channels[0], 0.0, batch_norm)
        self.encoder = nn.ModuleList(
            [
                Down(
                    self.channels[depth - 1],
                    self.channels[depth],
                    0.0 if depth == 1 else dropout,
                    batch_norm,
                )
                for depth in range(1, 5)
            ]
        )

        self.ups = nn.ModuleDict()
        self.nested_nodes = nn.ModuleDict()
        for stage in range(1, 5):
            for depth in range(5 - stage):
                key = self._node_key(depth, stage)
                self.ups[key] = nn.ConvTranspose2d(
                    self.channels[depth + 1],
                    self.channels[depth],
                    kernel_size=2,
                    stride=2,
                )
                node_dropout = dropout if depth >= 2 else 0.0
                self.nested_nodes[key] = DoubleConv(
                    (stage + 1) * self.channels[depth],
                    self.channels[depth],
                    node_dropout,
                    batch_norm,
                )
        self.outc = nn.Conv2d(self.channels[0], out_channels, kernel_size=1)

    @staticmethod
    def _node_key(depth: int, stage: int) -> str:
        return f"x{depth}_{stage}"

    @staticmethod
    def _align_to_skip(x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        diff_y = skip.size(2) - x.size(2)
        diff_x = skip.size(3) - x.size(3)
        if diff_x or diff_y:
            x = F.pad(
                x,
                [diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2],
            )
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        nodes: dict[tuple[int, int], torch.Tensor] = {(0, 0): self.inc(x)}
        for depth, down in enumerate(self.encoder, start=1):
            nodes[(depth, 0)] = down(nodes[(depth - 1, 0)])

        for stage in range(1, 5):
            for depth in range(5 - stage):
                key = self._node_key(depth, stage)
                upsampled = self.ups[key](nodes[(depth + 1, stage - 1)])
                upsampled = self._align_to_skip(upsampled, nodes[(depth, 0)])
                dense_features = [nodes[(depth, previous)] for previous in range(stage)]
                nodes[(depth, stage)] = self.nested_nodes[key](
                    torch.cat([*dense_features, upsampled], dim=1)
                )
        return self.outc(nodes[(0, 4)])


class ResidualConvBlock(nn.Module):
    """Two convolutions with an identity/projection shortcut."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dropout: float = 0.0,
        batch_norm: bool = True,
    ) -> None:
        super().__init__()

        def normalization() -> nn.Module:
            return nn.BatchNorm2d(out_channels) if batch_norm else nn.Identity()

        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            bias=not batch_norm,
        )
        self.norm1 = normalization()
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            bias=not batch_norm,
        )
        self.norm2 = normalization()
        if in_channels == out_channels:
            self.shortcut = nn.Identity()
        else:
            shortcut: list[nn.Module] = [
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=not batch_norm)
            ]
            if batch_norm:
                shortcut.append(nn.BatchNorm2d(out_channels))
            self.shortcut = nn.Sequential(*shortcut)
        self.activation = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)
        x = self.activation(self.norm1(self.conv1(x)))
        x = self.norm2(self.conv2(x))
        return self.dropout(self.activation(x + residual))


class ResidualDown(nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dropout: float = 0.0,
        batch_norm: bool = True,
    ) -> None:
        super().__init__(
            nn.MaxPool2d(2),
            ResidualConvBlock(in_channels, out_channels, dropout, batch_norm),
        )


class ResidualAttentionUp(nn.Module):
    """Residual decoder block with an attention-filtered skip connection."""

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
        self.gate = AttentionGate(
            gate_channels=out_channels,
            skip_channels=skip_channels,
            intermediate_channels=max(skip_channels // 2, 1),
            batch_norm=batch_norm,
        )
        self.conv = ResidualConvBlock(
            out_channels + skip_channels,
            out_channels,
            dropout,
            batch_norm,
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
        filtered_skip = self.gate(x, skip)
        return self.conv(torch.cat([filtered_skip, x], dim=1))


class ResidualAttentionASPPUNet(nn.Module):
    """Compact residual U-Net combining attention-filtered skips and ASPP context."""

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        base_channels: int = 24,
        dropout: float = 0.1,
        batch_norm: bool = True,
        aspp_branch_channels: int | None = None,
        aspp_dilation_rates: tuple[int, ...] = (1, 2, 4, 8),
        aspp_dropout: float | None = None,
    ) -> None:
        super().__init__()
        channels = [base_channels * (2**index) for index in range(5)]
        branch_channels = (
            max(channels[4] // 4, 1)
            if aspp_branch_channels is None
            else aspp_branch_channels
        )
        self.inc = ResidualConvBlock(in_channels, channels[0], 0.0, batch_norm)
        self.down1 = ResidualDown(channels[0], channels[1], 0.0, batch_norm)
        self.down2 = ResidualDown(channels[1], channels[2], dropout, batch_norm)
        self.down3 = ResidualDown(channels[2], channels[3], dropout, batch_norm)
        self.down4 = ResidualDown(channels[3], channels[4], dropout, batch_norm)
        self.aspp = ASPP(
            channels[4],
            channels[4],
            branch_channels,
            dilation_rates=aspp_dilation_rates,
            dropout=dropout if aspp_dropout is None else aspp_dropout,
            batch_norm=batch_norm,
        )
        self.up1 = ResidualAttentionUp(
            channels[4], channels[3], channels[3], dropout, batch_norm
        )
        self.up2 = ResidualAttentionUp(
            channels[3], channels[2], channels[2], dropout, batch_norm
        )
        self.up3 = ResidualAttentionUp(
            channels[2], channels[1], channels[1], 0.0, batch_norm
        )
        self.up4 = ResidualAttentionUp(
            channels[1], channels[0], channels[0], 0.0, batch_norm
        )
        self.outc = nn.Conv2d(channels[0], out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.aspp(self.down4(x4))
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
    if name in {
        "res_attention_aspp_unet",
        "resatt_aspp_unet",
        "resattasppunet",
    }:
        dilation_rates_value = config.get("aspp_dilation_rates", (1, 2, 4, 8))
        if not isinstance(dilation_rates_value, (list, tuple)):
            raise ValueError("model.aspp_dilation_rates must be a list or tuple")
        branch_channels_value = config.get("aspp_branch_channels")
        aspp_dropout_value = config.get("aspp_dropout")
        return ResidualAttentionASPPUNet(
            in_channels=int(config["in_channels"]),
            out_channels=int(config["out_channels"]),
            base_channels=int(config.get("base_channels", 24)),
            dropout=float(config.get("dropout", 0.0)),
            batch_norm=bool(config.get("batch_norm", True)),
            aspp_branch_channels=(
                int(branch_channels_value) if branch_channels_value is not None else None
            ),
            aspp_dilation_rates=tuple(int(rate) for rate in dilation_rates_value),
            aspp_dropout=float(aspp_dropout_value) if aspp_dropout_value is not None else None,
        )
    if name in {"aspp_unet", "asppunet", "unet_aspp"}:
        dilation_rates_value = config.get("aspp_dilation_rates", (1, 2, 4, 8))
        if not isinstance(dilation_rates_value, (list, tuple)):
            raise ValueError("model.aspp_dilation_rates must be a list or tuple")
        branch_channels_value = config.get("aspp_branch_channels")
        aspp_dropout_value = config.get("aspp_dropout")
        return ASPPUNet(
            in_channels=int(config["in_channels"]),
            out_channels=int(config["out_channels"]),
            base_channels=int(config["base_channels"]),
            dropout=float(config.get("dropout", 0.0)),
            batch_norm=bool(config.get("batch_norm", True)),
            aspp_branch_channels=(
                int(branch_channels_value) if branch_channels_value is not None else None
            ),
            aspp_dilation_rates=tuple(int(rate) for rate in dilation_rates_value),
            aspp_dropout=float(aspp_dropout_value) if aspp_dropout_value is not None else None,
        )
    if name in {"unet_plus_plus", "unetplusplus", "unet++"}:
        return UNetPlusPlus(
            in_channels=int(config["in_channels"]),
            out_channels=int(config["out_channels"]),
            base_channels=int(config["base_channels"]),
            dropout=float(config.get("dropout", 0.0)),
            batch_norm=bool(config.get("batch_norm", True)),
        )
    if name in {"attention_unet", "attentionunet", "att_unet"}:
        return AttentionUNet(
            in_channels=int(config["in_channels"]),
            out_channels=int(config["out_channels"]),
            base_channels=int(config["base_channels"]),
            dropout=float(config.get("dropout", 0.0)),
            batch_norm=bool(config.get("batch_norm", True)),
        )
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
