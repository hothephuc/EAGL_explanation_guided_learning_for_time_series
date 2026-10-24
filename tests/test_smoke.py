"""End-to-end smoke test on a tiny synthetic dataset (CPU, a few seconds)."""
import os

import numpy as np
import pandas as pd
import torch

from eagle import CLASS_NAMES
from eagle.data import CPSC2018Dataset, make_dataloaders
from eagle.engine.evaluator import deletion_insertion_auc
from eagle.engine.trainer import train_eagle
from eagle.models import Eagle, seresnet34

L = 2500


def make_fake_dataset(root, per_class=6, n_masked=4):
    mask_dir = os.path.join(root, 'mask')
    os.makedirs(mask_dir)
    rng = np.random.default_rng(0)
    rows = []
    for label in range(len(CLASS_NAMES)):
        for i in range(per_class):
            record = f'R{label}_{i}'
            np.savez(os.path.join(root, record + '.npz'), x=rng.random((12, L)).astype(np.float32))
            rows.append({'record': record, 'label': label})
    for i in range(n_masked):  # expert masks on PAC records
        mask = np.zeros(L, dtype=np.int8)
        mask[500:700] = 1
        np.savez(os.path.join(mask_dir, f'R5_{i}_mask.npz'), mask=mask)
    return pd.DataFrame(rows), mask_dir


def test_masks_are_loaded(tmp_path):
    meta, mask_dir = make_fake_dataset(str(tmp_path))
    ds = CPSC2018Dataset(meta, str(tmp_path), mask_dir=mask_dir)
    masked = meta.index[meta.record == 'R5_0'][0]
    unmasked = meta.index[meta.record == 'R0_0'][0]
    assert ds[masked][1].sum() == 200
    assert ds[unmasked][1].sum() == 0


def test_cache_matches_disk(tmp_path):
    meta, mask_dir = make_fake_dataset(str(tmp_path))
    disk = CPSC2018Dataset(meta, str(tmp_path), mask_dir=mask_dir)
    cached = CPSC2018Dataset(meta, str(tmp_path), mask_dir=mask_dir, cache=True)
    for i in range(len(meta)):
        for a, b in zip(disk[i], cached[i]):
            assert torch.equal(a, b)


def test_folds_are_disjoint(tmp_path):
    meta, mask_dir = make_fake_dataset(str(tmp_path))
    folds = make_dataloaders(meta, str(tmp_path), mask_dir=mask_dir, n_splits=3, batch_size=8,
                             num_workers=0, pin_memory=False, use_mask=True, n_mask_train=3)
    for _, _, train_df, val_df in folds:
        assert not set(train_df.record) & set(val_df.record)
    masked = {f'R5_{i}' for i in range(4)}
    val_unmasked = [r for _, _, _, val_df in folds for r in val_df.record if r not in masked]
    # every unmasked record lands in exactly one validation fold
    assert sorted(val_unmasked) == sorted(r for r in meta.record if r not in masked)


def test_train_and_explain(tmp_path):
    meta, mask_dir = make_fake_dataset(str(tmp_path))
    train_dl, val_dl, _, _ = make_dataloaders(meta, str(tmp_path), mask_dir=mask_dir, n_splits=2, batch_size=8,
                                              num_workers=0, pin_memory=False, use_mask=True, n_mask_train=3)[0]
    bmodel = seresnet34()
    model = Eagle(bmodel, target_layer=bmodel.layer4[-1].conv2)
    _, history, best_state, _ = train_eagle(model, train_dl, val_dl, device='cpu', num_epochs=1,
                                            use_gain=False, gain_epoch=1, gamma=10, threshold=0.4, omega=10)
    assert history['att_loss'][0] > 0  # expert masks actually contribute to the loss
    model.load_state_dict(best_state)

    x, _, _ = next(iter(val_dl))
    with torch.no_grad():
        _, logits, _, cam, _, _, _ = model(x, use_gain=True)
    assert cam.shape == (x.size(0), L)
    del_auc, ins_auc, *_ = deletion_insertion_auc(model, x, cam, logits.argmax(1))
    assert del_auc.shape == ins_auc.shape == (x.size(0),)
