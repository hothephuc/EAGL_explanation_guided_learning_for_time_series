"""Evaluate every trained fold and write a cross-fold summary with figures.

Usage: python tools/summarize_folds.py --config configs/default.yaml [--out docs/results]

Reads `results/<model_name>_fold{k}.pth`, `results/val_fold_{k}.csv` and `results/<model_name>_hist_fold_{k}.csv`
for every fold that exists, and writes to `--out`:
  metrics.json          per-fold and mean ± std macro metrics, per-class F1, Deletion / Insertion AUC
  per_class.csv         per-class precision / recall / F1 (mean ± std over folds)
  *.png                 figures (per-class F1, training curves, confusion matrix, explanation metrics)
"""
import argparse
import json
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, roc_auc_score
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from eagle import CLASS_NAMES  # noqa: E402
from eagle.config import load_config, resolve_device  # noqa: E402
from eagle.data import CPSC2018Dataset  # noqa: E402
from eagle.engine.evaluator import evaluate_del_ins_auc, get_predictions  # noqa: E402
from eagle.models import BasicBlock, Eagle, ResNet  # noqa: E402

# Thesis Table 4.5 (SE-ResNet34 + EGL, mean over 5 folds) and Table 4.7 / Fig. 4.3
THESIS_F1 = {'Normal': .8450, 'AFIB': .9238, '1AVB': .8894, 'LBBB': .9012, 'RBBB': .9344,
             'PAC': .7288, 'PVC': .8894, 'STD': .8316, 'STE': .7701}
THESIS_MACRO = {'precision': .8680, 'recall': .8488, 'f1': .8571}
THESIS_DELETION = {'PAC': .3374, 'PVC': .5241}
THESIS_INSERTION_OTHERS = .8426

BLUE, GREY, ORANGE = '#2a78d6', '#9aa0a6', '#eb6834'
plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.grid': True, 'grid.alpha': .25, 'savefig.dpi': 160, 'savefig.bbox': 'tight'})


def evaluate_fold(cfg, k, device):
    data_cfg, model_cfg, train_cfg = cfg['data'], cfg['model'], cfg['train']
    save_dir, name = train_cfg['save_dir'], train_cfg['model_name']
    val_df = pd.read_csv(os.path.join(save_dir, f'val_fold_{k}.csv'))
    ds = CPSC2018Dataset(val_df, data_cfg['db_dir'], cache=True)
    dl = DataLoader(ds, batch_size=cfg['eval']['batch_size'], shuffle=False)

    bmodel = ResNet(BasicBlock, num_blocks=model_cfg['num_blocks'], num_classes=model_cfg['num_classes'])
    model = Eagle(bmodel, num_classes=model_cfg['num_classes'], target_layer=bmodel.layer4[-1].conv2).to(device)
    model.load_state_dict(torch.load(os.path.join(save_dir, f'{name}_fold{k}.pth'), map_location=device))
    model.eval()

    y_true, y_pred, y_prob = get_predictions(model, dl, device)
    labels = np.arange(len(CLASS_NAMES))
    p, r, f, support = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    expl = evaluate_del_ins_auc(model, dl, device=device, class_names=CLASS_NAMES)

    per_class_del = {c: float(expl['del'][expl['labels'] == i].mean()) for i, c in enumerate(CLASS_NAMES)}
    per_class_ins = {c: float(expl['ins'][expl['labels'] == i].mean()) for i, c in enumerate(CLASS_NAMES)}
    others = ~np.isin(expl['labels'], [CLASS_NAMES.index('PAC'), CLASS_NAMES.index('PVC')])
    return {
        'fold': k,
        'n_val': int(len(y_true)),
        'macro': {'precision': float(p.mean()), 'recall': float(r.mean()), 'f1': float(f.mean()),
                  'accuracy': float((y_true == y_pred).mean()),
                  'roc_auc_ovr': float(roc_auc_score(y_true, y_prob, multi_class='ovr', average='macro'))},
        'per_class': {c: {'precision': float(p[i]), 'recall': float(r[i]), 'f1': float(f[i]), 'support': int(support[i])}
                      for i, c in enumerate(CLASS_NAMES)},
        'deletion': per_class_del,
        'insertion': per_class_ins,
        'insertion_other_classes': float(expl['ins'][others].mean()),
        'confusion_matrix': confusion_matrix(y_true, y_pred, labels=labels).tolist(),
    }


def mean_std(values):
    return {'mean': float(np.mean(values)), 'std': float(np.std(values))}


def summarize(folds):
    s = {'n_folds': len(folds),
         'macro': {m: mean_std([f['macro'][m] for f in folds]) for m in folds[0]['macro']},
         'per_class_f1': {c: mean_std([f['per_class'][c]['f1'] for f in folds]) for c in CLASS_NAMES},
         'deletion': {c: mean_std([f['deletion'][c] for f in folds]) for c in CLASS_NAMES},
         'insertion': {c: mean_std([f['insertion'][c] for f in folds]) for c in CLASS_NAMES},
         'insertion_other_classes': mean_std([f['insertion_other_classes'] for f in folds])}
    return s


def plot_per_class_f1(summary, out):
    x = np.arange(len(CLASS_NAMES)); w = .38
    ours = [summary['per_class_f1'][c]['mean'] for c in CLASS_NAMES]
    err = [summary['per_class_f1'][c]['std'] for c in CLASS_NAMES]
    fig, ax = plt.subplots(figsize=(9, 3.8))
    ax.bar(x - w / 2, [THESIS_F1[c] for c in CLASS_NAMES], w, color=GREY, label='Thesis (Table 4.5)')
    ax.bar(x + w / 2, ours, w, yerr=err, capsize=3, color=BLUE, label='This reproduction (mean ± std)')
    for i, v in enumerate(ours):
        ax.text(i + w / 2, v + err[i] + .01, f'{v:.3f}', ha='center', fontsize=8)
    ax.set_xticks(x, CLASS_NAMES); ax.set_ylim(.5, 1.02); ax.set_ylabel('F1-score')
    ax.set_title('Per-class F1, SE-ResNet34 + EGL', pad=28)
    ax.legend(loc='lower center', bbox_to_anchor=(.5, 1.0), ncol=2, frameon=False)
    fig.savefig(os.path.join(out, 'per_class_f1.png')); plt.close(fig)


def plot_training_curves(hists, out):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    for k, h in hists.items():
        e = np.arange(1, len(h) + 1)
        axes[0].plot(e, h['train_loss'], color=BLUE, alpha=.7, lw=1)
        axes[0].plot(e, h['val_loss'], color=ORANGE, alpha=.6, lw=1)
        axes[1].plot(e, h['train_acc'], color=BLUE, alpha=.7, lw=1)
        axes[1].plot(e, h['val_acc'], color=ORANGE, alpha=.6, lw=1)
    for ax, t in zip(axes, ['Loss', 'Accuracy (%)']):
        ax.set_xlabel('epoch'); ax.set_title(t)
        ax.plot([], [], color=BLUE, label='train'); ax.plot([], [], color=ORANGE, label='validation')
        ax.legend(frameon=False)
    fig.suptitle(f'Training curves, {len(hists)} fold{"s" if len(hists) > 1 else ""} (one line per fold)', y=1.03)
    fig.savefig(os.path.join(out, 'training_curves.png')); plt.close(fig)


def plot_confusion(folds, out):
    cm = np.sum([np.array(f['confusion_matrix']) for f in folds], axis=0)
    cmn = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(6.4, 5.4))
    im = ax.imshow(cmn, cmap='Blues', vmin=0, vmax=1)
    for i in range(len(cmn)):
        for j in range(len(cmn)):
            ax.text(j, i, f'{cmn[i, j]:.2f}', ha='center', va='center', fontsize=8,
                    color='white' if cmn[i, j] > .55 else '#222')
    ax.set_xticks(range(len(CLASS_NAMES)), CLASS_NAMES, rotation=45); ax.set_yticks(range(len(CLASS_NAMES)), CLASS_NAMES)
    ax.set_xlabel('Predicted'); ax.set_ylabel('True'); ax.grid(False)
    ax.set_title(f'Confusion matrix (row-normalised, {len(folds)} fold{"s" if len(folds) > 1 else ""} pooled)')
    fig.colorbar(im, ax=ax, fraction=.046)
    fig.savefig(os.path.join(out, 'confusion_matrix.png')); plt.close(fig)


def plot_explanations(summary, out):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    cls = ['PAC', 'PVC']; x = np.arange(2); w = .38
    ours = [summary['deletion'][c]['mean'] for c in cls]; err = [summary['deletion'][c]['std'] for c in cls]
    axes[0].bar(x - w / 2, [THESIS_DELETION[c] for c in cls], w, color=GREY, label='Thesis')
    axes[0].bar(x + w / 2, ours, w, yerr=err, capsize=3, color=BLUE, label='Reproduction')
    axes[0].set_xticks(x, cls); axes[0].set_ylabel('AUC'); axes[0].set_title('Deletion AUC on masked classes (lower is better)')
    axes[0].legend(frameon=False)
    x = np.arange(len(CLASS_NAMES))
    ins = [summary['insertion'][c]['mean'] for c in CLASS_NAMES]; dele = [summary['deletion'][c]['mean'] for c in CLASS_NAMES]
    axes[1].bar(x - w / 2, ins, w, color=BLUE, label='Insertion ↑')
    axes[1].bar(x + w / 2, dele, w, color=ORANGE, label='Deletion ↓')
    axes[1].set_xticks(x, CLASS_NAMES, rotation=45); axes[1].set_ylim(0, 1)
    axes[1].set_title('Explanation faithfulness per class'); axes[1].legend(frameon=False)
    fig.savefig(os.path.join(out, 'explanation_metrics.png')); plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--config', default='configs/default.yaml')
    parser.add_argument('--out', default='docs/results')
    args = parser.parse_args()
    cfg = load_config(args.config)
    device = resolve_device(cfg['model'].get('device'))
    save_dir, name = cfg['train']['save_dir'], cfg['train']['model_name']
    ks = [k for k in range(1, cfg['train']['n_splits'] + 1) if os.path.exists(os.path.join(save_dir, f'{name}_fold{k}.pth'))]
    if not ks:
        sys.exit(f'No checkpoints found in {save_dir}/')
    os.makedirs(args.out, exist_ok=True)

    folds = []
    for k in ks:
        print(f'\n===== Evaluating fold {k} =====')
        folds.append(evaluate_fold(cfg, k, device))
    summary = summarize(folds)
    hists = {k: pd.read_csv(os.path.join(save_dir, f'{name}_hist_fold_{k}.csv')) for k in ks}

    with open(os.path.join(args.out, 'metrics.json'), 'w') as fh:
        json.dump({'summary': summary, 'thesis': {'macro': THESIS_MACRO, 'per_class_f1': THESIS_F1,
                   'deletion': THESIS_DELETION, 'insertion_other_classes': THESIS_INSERTION_OTHERS},
                   'folds': folds, 'config': cfg}, fh, indent=2)
    rows = []
    for c in CLASS_NAMES:
        row = {'class': c}
        for m in ('precision', 'recall', 'f1'):
            v = [f['per_class'][c][m] for f in folds]
            row[m] = f'{np.mean(v):.4f} ± {np.std(v):.4f}'
        row['thesis_f1'] = THESIS_F1[c]
        rows.append(row)
    pd.DataFrame(rows).to_csv(os.path.join(args.out, 'per_class.csv'), index=False)

    plot_per_class_f1(summary, args.out)
    plot_training_curves(hists, args.out)
    plot_confusion(folds, args.out)
    plot_explanations(summary, args.out)

    m = summary['macro']
    print(f"\nMacro over {len(ks)} folds: P {m['precision']['mean']:.4f}±{m['precision']['std']:.4f} | "
          f"R {m['recall']['mean']:.4f}±{m['recall']['std']:.4f} | F1 {m['f1']['mean']:.4f}±{m['f1']['std']:.4f} "
          f"(thesis F1 {THESIS_MACRO['f1']})")
    print(f'Wrote {args.out}/')


if __name__ == '__main__':
    main()
