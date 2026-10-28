import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock2D(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=0.0):
        super().__init__()
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]
        if dropout > 0:
            layers.append(nn.Dropout2d(dropout))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class DownBlock2D(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=0.0):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = ConvBlock2D(in_channels, out_channels, dropout=dropout)

    def forward(self, x):
        return self.conv(self.pool(x))


class UpBlock2D(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels, dropout=0.0):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = ConvBlock2D(out_channels + skip_channels, out_channels, dropout=dropout)

    def forward(self, x, skip):
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


class UNet2DEncoder(nn.Module):
    def __init__(self, in_channels=3, base_channels=32, has_dropout=True):
        super().__init__()
        dropout = 0.1 if has_dropout else 0.0
        self.in_conv = ConvBlock2D(in_channels, base_channels)
        self.down1 = DownBlock2D(base_channels, base_channels * 2)
        self.down2 = DownBlock2D(base_channels * 2, base_channels * 4)
        self.down3 = DownBlock2D(base_channels * 4, base_channels * 8, dropout=dropout)
        self.down4 = DownBlock2D(base_channels * 8, base_channels * 16, dropout=dropout)

    def forward(self, x):
        x1 = self.in_conv(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        return [x1, x2, x3, x4, x5]


class UNet2DDecoder(nn.Module):
    def __init__(self, n_classes=2, base_channels=32, has_dropout=True):
        super().__init__()
        dropout = 0.1 if has_dropout else 0.0
        self.up1 = UpBlock2D(base_channels * 16, base_channels * 8, base_channels * 8, dropout=dropout)
        self.up2 = UpBlock2D(base_channels * 8, base_channels * 4, base_channels * 4)
        self.up3 = UpBlock2D(base_channels * 4, base_channels * 2, base_channels * 2)
        self.up4 = UpBlock2D(base_channels * 2, base_channels, base_channels)
        self.out_conv = nn.Conv2d(base_channels, n_classes, kernel_size=1)

    def forward(self, features):
        x1, x2, x3, x4, x5 = features
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        feat = self.up4(x, x1)
        logits = self.out_conv(feat)
        return logits, feat


class UNet2D(nn.Module):
    def __init__(self, n_channels=3, n_classes=2, base_channels=32, has_dropout=True):
        super().__init__()
        self.encoder = UNet2DEncoder(
            in_channels=n_channels,
            base_channels=base_channels,
            has_dropout=has_dropout,
        )
        self.decoder = UNet2DDecoder(
            n_classes=n_classes,
            base_channels=base_channels,
            has_dropout=has_dropout,
        )

    def forward(self, x):
        logits, _ = self.decoder(self.encoder(x))
        return logits


class Mine2dUNetV1(nn.Module):
    """Shared-encoder, two-decoder 2D U-Net for PSTS/RCCR training."""

    def __init__(self, n_channels=3, n_classes=2, base_channels=32, has_dropout=True):
        super().__init__()
        self.encoder = UNet2DEncoder(
            in_channels=n_channels,
            base_channels=base_channels,
            has_dropout=has_dropout,
        )
        self.decoder1 = UNet2DDecoder(
            n_classes=n_classes,
            base_channels=base_channels,
            has_dropout=has_dropout,
        )
        self.decoder2 = UNet2DDecoder(
            n_classes=n_classes,
            base_channels=base_channels,
            has_dropout=has_dropout,
        )

    def forward(self, x):
        features = self.encoder(x)
        out1, feat1 = self.decoder1(features)
        out2, feat2 = self.decoder2(features)
        if self.training:
            return [out1, out2], [feat1, feat2]
        return out1, out2
