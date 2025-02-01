import torch
from torch.utils.data import DataLoader, Dataset

class SignalDataset(Dataset):
    def __init__(self, X, y, transform=None):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)
        self.transform = transform
        
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        signal = self.X[idx]
        label = self.y[idx]

        if self.transform:
            signal = self.transform(signal)
            
        return signal, label

def create_signal_dataloader(X, y, batch_size=32, num_workers=4, shuffle=True, transform=None):
    dataset = SignalDataset(X, y, transform=transform)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        # num_workers=num_workers,
        pin_memory=True
    )