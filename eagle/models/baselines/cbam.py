import torch
import torch.nn as nn
import torch.nn.functional as F

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)

        self.fc = nn.Sequential(
            nn.Conv1d(in_planes, in_planes // reduction, 1, bias=False),
            nn.ReLU(),
            nn.Conv1d(in_planes // reduction, in_planes, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv1d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        attention = self.conv(x_cat)
        return self.sigmoid(attention)

class CBAM(nn.Module):
    def __init__(self, channels, reduction=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(channels, reduction)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        x = x * self.channel_attention(x)
        x = x * self.spatial_attention(x)
        return x

# --- BASIC BLOCK WITH CBAM ---
class BasicBlockCBAM(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1,
                 reduction=16, drop_connect_rate=0.0):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.cbam = CBAM(out_channels, reduction=reduction)
        self.relu = nn.ReLU(inplace=True)
        self.drop_connect_rate = drop_connect_rate

        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm1d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.cbam(out)
        if self.drop_connect_rate:
            out = F.dropout(out, p=self.drop_connect_rate, training=self.training)
        out = out + self.shortcut(x)
        return self.relu(out)

# --- RESNET WITH CBAM ---
class ResNetCBAM(nn.Module):
    def __init__(self, block, num_blocks, num_classes=9, drop_connect_rate=0.2):
        super(ResNetCBAM, self).__init__()
        self.in_channels = 64

        self.conv1 = nn.Conv1d(12, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.AvgPool1d(kernel_size=3, stride=2, padding=1)
        self.dropout = nn.Dropout(drop_connect_rate)
        self.lead_attention = CBAM(12, reduction=1)

        # ResNet stages
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1, drop_connect_rate=drop_connect_rate)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2, drop_connect_rate=drop_connect_rate)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2, drop_connect_rate=drop_connect_rate)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2, drop_connect_rate=drop_connect_rate)

        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, out_channels, num_blocks, stride, drop_connect_rate):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for i, s in enumerate(strides):
            layers.append(block(self.in_channels, out_channels, s, drop_connect_rate=drop_connect_rate))
            self.in_channels = out_channels * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        # Lead Attention (before first conv)
        x = self.lead_attention(x)

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.maxpool(out)

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        return out