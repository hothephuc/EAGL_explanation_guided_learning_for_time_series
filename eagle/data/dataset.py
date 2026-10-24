import os

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm

MASK_SUFFIX = '_mask'


def list_masked_records(mask_dir):
    """Return the set of record names that have an expert mask in `mask_dir`."""
    if not os.path.isdir(mask_dir):
        return set()
    return {
        os.path.splitext(f)[0][:-len(MASK_SUFFIX)]
        for f in os.listdir(mask_dir)
        if f.endswith(MASK_SUFFIX + '.npz')
    }


class CPSC2018Dataset(Dataset):
    """Pre-processed CPSC2018 windows.

    Each item is `(signal, mask, label)`:
      - signal: (12, L) float tensor loaded from `<data_dir>/<record>.npz` (key `x`)
      - mask:   (L,) float tensor loaded from `<mask_dir>/<record>_mask.npz` (key `mask`),
                or all zeros when the record has no expert mask
      - label:  integer class index

    With `cache=True` every window is read once into memory (~120 KB per window as float32),
    so later epochs do no disk I/O; use it with `num_workers=0`.
    """

    def __init__(self, meta_df, data_dir, transform=None, mask_dir=None, cache=False):
        self.df = meta_df.reset_index(drop=True)
        self.labels = self.df['label'].to_numpy()
        self.data_dir = data_dir
        self.transform = transform
        self.mask_dir = mask_dir
        self.masked_records = list_masked_records(mask_dir) if mask_dir else set()
        self.signals = self.masks = None
        if cache:
            self._preload()

    def _load(self, record):
        signal = np.load(os.path.join(self.data_dir, record + '.npz'))['x']  # (12, length)
        if record in self.masked_records:
            mask = np.load(os.path.join(self.mask_dir, record + MASK_SUFFIX + '.npz'))['mask']
        else:
            mask = np.zeros(signal.shape[-1])
        return signal, mask

    def _preload(self):
        first, _ = self._load(self.df.record[0])
        signals = np.empty((len(self.df),) + first.shape, dtype=np.float32)
        masks = np.empty((len(self.df), first.shape[-1]), dtype=np.float32)
        for i, record in enumerate(tqdm(self.df.record, desc='Caching dataset in memory')):
            signals[i], masks[i] = self._load(record)
        self.signals, self.masks = torch.from_numpy(signals), torch.from_numpy(masks)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if self.signals is not None:
            signal, mask = self.signals[idx], self.masks[idx]
        else:
            signal, mask = self._load(self.df.record[idx])
        if self.transform:
            signal = self.transform(signal)

        signal = torch.as_tensor(signal, dtype=torch.float32)
        mask = torch.as_tensor(mask, dtype=torch.float32)
        labels = torch.tensor(self.labels[idx], dtype=torch.long)
        return signal, mask, labels


def dataset_labels(dataset):
    """Labels of a `CPSC2018Dataset` or a `Subset` of one, without loading any signal."""
    if isinstance(dataset, Subset):
        return dataset_labels(dataset.dataset)[np.asarray(dataset.indices)]
    return dataset.labels


def make_dataloaders(meta_data, data_dir, mask_dir=None, n_splits=5, batch_size=64, num_workers=4,
                     pin_memory=True, seed=1886, use_mask=False, n_mask_train=400, cache=False):
    """Build stratified k-fold train/val loaders.

    With `use_mask=True`, the records that have an expert mask are kept out of the k-fold split:
    `n_mask_train` of them are added to every training fold and the rest to every validation fold.

    All folds share one underlying dataset, so with `cache=True` the data is read into memory only once.
    """
    meta_data = meta_data.reset_index(drop=True)
    masked_records = list_masked_records(mask_dir) if (use_mask and mask_dir) else set()
    is_masked = meta_data['record'].isin(masked_records)
    unmask_data = meta_data[~is_masked]
    mask_data = meta_data[is_masked].sample(frac=1, random_state=seed)

    full_ds = CPSC2018Dataset(meta_data, data_dir, mask_dir=mask_dir if use_mask else None, cache=cache)
    if cache:
        num_workers = 0  # indexing in-memory tensors is faster than shipping them to worker processes
    loader_kwargs = dict(batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory,
                         persistent_workers=num_workers > 0)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    folds = []
    for train_idx, val_idx in skf.split(unmask_data, unmask_data['label'].values):
        train_rows = np.concatenate([unmask_data.index[train_idx], mask_data.index[:n_mask_train]])
        val_rows = np.concatenate([unmask_data.index[val_idx], mask_data.index[n_mask_train:]])
        train_df = meta_data.loc[train_rows].reset_index(drop=True)
        val_df = meta_data.loc[val_rows].reset_index(drop=True)

        train_loader = DataLoader(Subset(full_ds, train_rows), shuffle=True, **loader_kwargs)
        val_loader = DataLoader(Subset(full_ds, val_rows), shuffle=False, **loader_kwargs)
        folds.append((train_loader, val_loader, train_df, val_df))
    return folds
