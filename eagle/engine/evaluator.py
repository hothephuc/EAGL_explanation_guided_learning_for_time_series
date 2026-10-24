from typing import List, Optional, Tuple

import matplotlib.collections as mcoll
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn.functional as F
from matplotlib.colors import Normalize
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from tqdm import tqdm


def multicolored_lines(signals, heatmap, title, leads=(0, 1, 6), fs=250, seconds=10):
    """Plot a few ECG leads coloured by a saliency heatmap."""
    fig, ax = plt.subplots(figsize=(25, 5), nrows=len(leads), ncols=1, sharex=True)
    norm = Normalize(vmin=np.min(heatmap), vmax=np.max(heatmap))
    for i, lead in enumerate(leads):
        signal = signals[lead][:fs * seconds]
        points = np.array([np.arange(len(signal)), signal]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        lc = mcoll.LineCollection(segments, cmap='viridis', norm=norm)
        lc.set_array(heatmap[:len(signal) - 1])  # Set colors for each segment
        line = ax[i].add_collection(lc)

        ax[i].set_xlim(0, len(signal))
        ax[i].set_ylim(np.min(signal) - 0.1 * np.abs(np.min(signal)),
                       np.max(signal) + 0.1 * np.abs(np.max(signal)))
        ax[i].set_yticks([])

    fig.colorbar(line, ax=ax.ravel().tolist(), label='Heatmap Value')
    fig.supxlabel('Sample Index')
    fig.supylabel('Signal Value')
    fig.suptitle(title)
    return fig


def deletion_insertion_auc(model, x, saliency, target, baseline=0.0):
    """Deletion / Insertion AUC (Petsiuk et al., 2018) on time steps ranked by saliency.

    For each saliency threshold q (from 1 to -1):
      - deletion input:  time steps with saliency > q are replaced by `baseline`
      - insertion input: only time steps with saliency > q are kept, the rest is `baseline`
    """
    N, C, L = x.shape
    device = x.device

    thresholds = torch.tensor([1.0, 0.75, 0.5, 0.25, -1.0], device=device)
    del_probs, ins_probs = [], []

    model.eval()
    with torch.no_grad():
        for q in thresholds:
            low_saliency = (saliency <= q.item()).unsqueeze(1).expand(N, C, L)  # (N, 12, L)
            base = baseline if isinstance(baseline, torch.Tensor) else torch.full_like(x, baseline)

            deleted_x = torch.where(low_saliency, x, base)
            inserted_x = torch.where(low_saliency, base, x)

            p_del = F.softmax(model.bmodel(deleted_x), dim=1)[torch.arange(N), target]
            p_ins = F.softmax(model.bmodel(inserted_x), dim=1)[torch.arange(N), target]

            del_probs.append(p_del)
            ins_probs.append(p_ins)

    del_probs = torch.stack(del_probs, dim=0)  # (steps, N)
    ins_probs = torch.stack(ins_probs, dim=0)  # (steps, N)
    ins_del_probs = ins_probs - del_probs

    xs = torch.linspace(0, 1, thresholds.shape[0], device=device)
    del_auc = torch.trapz(del_probs, xs, dim=0)  # (N,)
    ins_auc = torch.trapz(ins_probs, xs, dim=0)  # (N,)
    ins_del_auc = torch.trapz(ins_del_probs, xs, dim=0)

    return (del_auc.cpu(), ins_auc.cpu(), ins_del_auc.cpu(),
            del_probs.cpu().numpy(), ins_probs.cpu().numpy(), ins_del_probs.cpu().numpy())


def plot_saliency_and_curves(x, saliency, deletion_scores, insertion_scores):
    B, C, L = x.shape
    for b in range(B):
        multicolored_lines(x[b].cpu().detach().numpy(), saliency[b].cpu().detach().numpy(), title=f'Sample {b+1}')

        steps = deletion_scores.shape[0]
        xs = torch.linspace(0, 1, steps)
        plt.figure()
        plt.plot(xs, deletion_scores[:, b], label='Deletion')
        plt.plot(xs, insertion_scores[:, b], label='Insertion')
        plt.legend()
        plt.title(f'Sample {b+1}: Deletion/Insertion Curves')
        plt.xlabel('Fraction of points perturbed')
        plt.ylabel('Model output')
    plt.show()


def get_predictions(
    model: torch.nn.Module,
    test_loader: torch.utils.data.DataLoader,
    device: str = 'cuda',
    return_prob=True
) -> Tuple[np.ndarray, ...]:

    model.eval()

    y_true = []
    y_pred = []
    y_prob = []
    with torch.no_grad():
        for signals, _, labels in test_loader:
            outputs = model.bmodel(signals.to(device))
            probs = torch.softmax(outputs, dim=1)
            y_true.extend(labels.numpy())
            y_pred.extend(outputs.argmax(1).cpu().numpy())
            y_prob.extend(probs.cpu().numpy())

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_prob = np.array(y_prob)
    if return_prob:
        return y_true, y_pred, y_prob
    return y_true, y_pred


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Optional[List[str]] = None,
    save_path: Optional[str] = None
) -> None:

    cm = confusion_matrix(y_true, y_pred, labels=np.arange(len(class_names)) if class_names else None)
    cm_normalized = cm.astype('float') / np.maximum(cm.sum(axis=1)[:, np.newaxis], 1)

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm_normalized,
        annot=True,
        fmt='.2f',
        cmap='Blues',
        xticklabels=class_names if class_names else 'auto',
        yticklabels=class_names if class_names else 'auto'
    )
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Confusion matrix saved to {save_path}")
    else:
        plt.show()
    plt.close()


def evaluate_del_ins_auc(
    model: torch.nn.Module,
    test_loader: torch.utils.data.DataLoader,
    device: str = 'cuda',
    class_names: Optional[List[str]] = None,
) -> dict:
    """Mean Deletion / Insertion AUC of the CAM explanations, overall and per true class."""
    model.eval()
    per_sample = {'label': [], 'del': [], 'ins': [], 'ins_del': []}

    for x, _, y in tqdm(test_loader, desc="Evaluating AUC"):
        x = x.to(device)
        with torch.no_grad():
            _, logits, _, sal, _, _, _ = model(x, use_gain=True)
        target = torch.argmax(logits, dim=1)
        del_auc, ins_auc, ins_del_auc, *_ = deletion_insertion_auc(model=model, x=x, saliency=sal, target=target)
        per_sample['label'].append(y)
        per_sample['del'].append(del_auc)
        per_sample['ins'].append(ins_auc)
        per_sample['ins_del'].append(ins_del_auc)

    labels = torch.cat(per_sample['label']).numpy()
    scores = {k: torch.cat(v).numpy() for k, v in per_sample.items() if k != 'label'}

    print(f"Mean Deletion AUC: {scores['del'].mean():.4f}")
    print(f"Mean Insertion AUC: {scores['ins'].mean():.4f}")
    print(f"Mean Insertion-Deletion AUC: {scores['ins_del'].mean():.4f}")

    print("\nPer-class Deletion (lower is better) / Insertion (higher is better) AUC:")
    print("-" * 60)
    for c in np.unique(labels):
        name = class_names[c] if class_names else f"Class {c}"
        sel = labels == c
        print(f"{name:>8}: Del {scores['del'][sel].mean():.4f} | Ins {scores['ins'][sel].mean():.4f} | n={sel.sum()}")
    return {'labels': labels, **scores}


def evaluate_model(
    model: torch.nn.Module,
    test_loader: torch.utils.data.DataLoader,
    class_names: Optional[List[str]] = None,
    device: str = 'cuda',
    save_path: Optional[str] = None
) -> None:

    y_true, y_pred, y_prob = get_predictions(model, test_loader, device, return_prob=True)

    print("\nClassification Report:")
    print("-" * 60)
    print(classification_report(
        y_true,
        y_pred,
        labels=np.arange(len(class_names)) if class_names else None,
        target_names=class_names,
        digits=4,
        zero_division=0
    ))

    cm = confusion_matrix(y_true, y_pred)
    per_class_accuracy = cm.diagonal() / np.maximum(cm.sum(axis=1), 1)

    print("\nPer-Class Accuracy:")
    print("-" * 60)
    for i, acc in enumerate(per_class_accuracy):
        class_label = class_names[i] if class_names else f"Class {i}"
        print(f"{class_label}: {acc:.4f}")

    try:
        if len(np.unique(y_true)) == 2:
            roc_auc = roc_auc_score(y_true, y_prob[:, 1])
        else:
            roc_auc = roc_auc_score(y_true, y_prob, multi_class='ovr', average='macro')
        print(f"\nROC AUC Score: {roc_auc:.4f}")
    except Exception as e:
        print(f"\nCould not calculate ROC AUC: {e}")

    plot_confusion_matrix(y_true, y_pred, class_names, save_path=save_path)
