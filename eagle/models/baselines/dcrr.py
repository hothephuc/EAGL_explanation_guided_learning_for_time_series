import torch
from torch.utils.data import Dataset
import numpy as np
from torch import nn
import torch.nn.functional as F
from scipy.signal import butter, filtfilt, find_peaks


def bandpass_filter(signal, lowcut=5, highcut=15, fs=250, order=1):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, signal)


def detect_r_peaks(signal, fs=250):
    # Pan-Tompkins simplified
    filtered = bandpass_filter(signal, 5, 15, fs)
    diffed = np.gradient(filtered)
    squared = diffed ** 2
    integrated = np.convolve(squared, np.ones(int(0.15 * fs)) / (0.15 * fs), mode='same')
    threshold = np.mean(integrated)
    peaks, _ = find_peaks(integrated, height=threshold, distance=int(0.2 * fs))
    return peaks


def compute_rr_intervals(r_peaks):
    if len(r_peaks) < 2:
        return {'max': 125, 'avg': 125, 'min': 125}  # fallback for short signals
    rr = np.diff(r_peaks)
    return {
        'max': int(np.max(rr)),
        'avg': int(np.mean(rr)),
        'min': int(np.min(rr))
    }
class ECGWithRRDataset(Dataset):
    def __init__(self, base_dataset, fs=250):
        """
        base_dataset: torch Dataset that returns (ecg_tensor, label)
        fs: sampling frequency of ECG (default 500 Hz)
        """
        self.base_dataset = base_dataset
        self.fs = fs

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        ecg, _, label = self.base_dataset[idx]  # ecg: (12, L), label: int or tensor

        lead_ii = ecg[1].numpy() if isinstance(ecg, torch.Tensor) else ecg[1]
        r_peaks = detect_r_peaks(lead_ii, fs=self.fs)
        rr_dict = compute_rr_intervals(r_peaks)

        rr_lengths = {k: torch.tensor(rr_dict[k], dtype=torch.int32) for k in ['max', 'avg', 'min']}
        return ecg, rr_lengths, label
class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_sizes=[1, 3, 5]):
        super().__init__()
        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(in_channels, in_channels, k, padding=k // 2, groups=in_channels),
                nn.Conv1d(in_channels, out_channels, 1),
                nn.BatchNorm1d(out_channels),
                nn.ReLU()
            )
            for k in kernel_sizes
        ])

    def forward(self, x):
        return sum(conv(x) for conv in self.convs)

class RRAttention(nn.Module):
    def __init__(self, channels, reduction=32):
        super().__init__()
        self.reduction = reduction
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.conv1 = nn.Conv1d(channels * 2, channels // reduction, kernel_size=1)
        self.conv2x = nn.Conv1d(channels // reduction, channels, kernel_size=1)
        self.conv2y = nn.Conv1d(channels // reduction, channels, kernel_size=1)

    def forward(self, x, rr_lengths):
        B, C, L = x.shape
        pooled_rr = []
        for rr_key in ['max', 'avg', 'min']:
            rr_lens = rr_lengths[rr_key]
            pooled = []
            for i in range(B):
                n = max(1, L // rr_lens[i].item())
                s = max(1, L // n)
                p = F.avg_pool1d(x[i:i+1], kernel_size=s, stride=s, ceil_mode=True)
                up = F.interpolate(p, size=L, mode='linear', align_corners=False)
                pooled.append(up)
            pooled_cat = torch.cat(pooled, dim=0)
            pooled_rr.append(pooled_cat)
        RX = sum(pooled_rr)
        RY = torch.mean(x, dim=2, keepdim=True).expand(-1, -1, L)
        z = self.relu(self.conv1(torch.cat([RX, RY], dim=1)))
        WX = self.sigmoid(self.conv2x(z))
        WY = self.sigmoid(self.conv2y(z))
        return x * WX * WY

class BranchNet(nn.Module):
    def __init__(self, in_channels, use_rr_attention=True):
        super().__init__()
        self.use_rr_attention = use_rr_attention

        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU()
        )

        self.dsc = nn.Sequential(
            DepthwiseSeparableConv(64, 128),
            nn.BatchNorm1d(128),
            nn.ReLU()
        )

        if use_rr_attention:
            self.rr_attention = RRAttention(128)

        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x, rr_lengths):
        x = self.conv1(x)
        x = self.dsc(x)
        if self.use_rr_attention:
            x = self.rr_attention(x, rr_lengths)
        return self.pool(x).squeeze(-1)


class FusionModule(nn.Module):
    def __init__(self, use_rr_attention=True):
        super().__init__()
        self.use_rr_attention = use_rr_attention
        if use_rr_attention:
            self.rr_attention = RRAttention(256)

    def forward(self, limb_feat, prec_feat, rr_lengths):
        def entropy(x):
            p = torch.softmax(x, dim=1)
            return -(p * torch.log(p + 1e-6)).sum(dim=1)

        ent_limb = entropy(limb_feat)
        ent_prec = entropy(prec_feat)
        weight = torch.softmax(torch.stack([ent_limb, ent_prec], dim=1), dim=1)

        combined = torch.cat([
            (1 + weight[:, 0].unsqueeze(1)) * limb_feat,
            (1 + weight[:, 1].unsqueeze(1)) * prec_feat
        ], dim=1).unsqueeze(-1)

        if self.use_rr_attention:
            combined = self.rr_attention(combined, rr_lengths)

        return combined.squeeze(-1)

class DCRRNet(nn.Module):
    def __init__(self, num_classes, use_rr_attention=True):
        super().__init__()
        self.limb_branch = BranchNet(6, use_rr_attention)
        self.prec_branch = BranchNet(6, use_rr_attention)
        self.fusion = FusionModule(use_rr_attention)
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x, rr_lengths):
        limb = x[:, :6, :]
        prec = x[:, 6:, :]

        limb_feat = self.limb_branch(limb, rr_lengths)
        prec_feat = self.prec_branch(prec, rr_lengths)
        fused_feat = self.fusion(limb_feat, prec_feat, rr_lengths)

        return self.classifier(fused_feat)