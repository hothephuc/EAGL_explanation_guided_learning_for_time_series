from torch import nn
import torch.nn.functional as F

class PreActBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.bn1 = nn.BatchNorm1d(in_channels)
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)

        self.bn2 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)

        self.skip = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.skip = nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)

    def forward(self, x):
        identity = self.skip(x)
        out = self.conv1(F.relu(self.bn1(x)))
        out = self.conv2(F.relu(self.bn2(out)))
        return out + identity


class XResNet1D(nn.Module):
    def __init__(self, in_channels=12, num_classes=9, layers=[2, 2, 2, 2]):
        super().__init__()

        # Stem with 3 convs
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(32),
            nn.ReLU(),

            nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(),

            nn.Conv1d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
        )

        # Residual stages
        self.inplanes = 64
        self.layer1 = self._make_layer(64, layers[0], stride=1)
        self.layer2 = self._make_layer(128, layers[1], stride=2)
        self.layer3 = self._make_layer(256, layers[2], stride=2)
        self.layer4 = self._make_layer(512, layers[3], stride=2)

        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, num_classes),
        )

    def _make_layer(self, out_channels, blocks, stride):
        layers = [PreActBlock1D(self.inplanes, out_channels, stride)]
        self.inplanes = out_channels
        for _ in range(1, blocks):
            layers.append(PreActBlock1D(self.inplanes, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stem(x)      # -> (B, 64, L/2)
        x = self.layer1(x)    # -> (B, 64, L/2)
        x = self.layer2(x)    # -> (B, 128, L/4)
        x = self.layer3(x)    # -> (B, 256, L/8)
        x = self.layer4(x)    # -> (B, 512, L/16)
        x = self.pool(x)      # -> (B, 512, 1)
        return self.fc(x)     # -> (B, num_classes)