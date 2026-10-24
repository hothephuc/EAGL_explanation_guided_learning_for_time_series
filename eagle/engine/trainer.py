import contextlib
import copy

import numpy as np
import torch
from sklearn.metrics import classification_report
from sklearn.utils.class_weight import compute_class_weight
from torch import optim
from tqdm import tqdm

from eagle.data import dataset_labels


def compute_class_weights(train_loader, num_classes, device):
    """Balanced class weights with the manual LBBB/PAC adjustment used in the thesis."""
    try:
        all_labels = dataset_labels(train_loader.dataset)
    except AttributeError:  # any other dataset: read the labels batch by batch
        all_labels = np.concatenate([labels.numpy() for _, _, labels in train_loader])
    class_weights_np = compute_class_weight(class_weight='balanced', classes=np.arange(num_classes), y=all_labels)
    # Manual adjustment from the thesis experiments: down-weight LBBB, up-weight PAC.
    class_weights_np[3] -= 1
    class_weights_np[5] += 1
    return torch.tensor(class_weights_np, dtype=torch.float, device=device)


def autocast(device, amp):
    """bfloat16 mixed precision when `amp == 'bf16'`, otherwise a no-op (full fp32)."""
    if amp == 'bf16':
        return torch.autocast(device_type=device.split(':')[0], dtype=torch.bfloat16)
    return contextlib.nullcontext()


def train_eagle(model, train_loader, val_loader, device='cuda', num_classes=9, use_gain=True, learning_rate=0.0001,
                weight_decay=1e-4, num_epochs=70, threshold=0.5, omega=10, gamma=0.1, gain_epoch=None, amp=None):
    """Train an `Eagle` model and keep the weights with the best validation accuracy.

    `gain_epoch` switches the explanation loss on from that (1-indexed) epoch onwards;
    `None` keeps `use_gain` unchanged for the whole run.
    `amp='bf16'` runs forward passes in bfloat16 mixed precision (~1.4x faster on Apple GPUs).
    """
    model = model.to(device)
    class_weights = compute_class_weights(train_loader, num_classes, device)

    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'cl_loss': [],
        'att_loss': [],
        'val_acc': []
    }

    best_val_acc = 0
    best_state = copy.deepcopy(model.state_dict())

    for epoch in range(num_epochs):
        # Training phase
        model.bmodel.train()
        train_loss = 0
        correct = 0
        total = 0
        train_att_loss = 0
        train_cl_loss = 0

        if gain_epoch is not None and epoch + 1 == gain_epoch:
            use_gain = True

        for signals, masks, labels in tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}'):
            signals, masks, labels = signals.to(device), masks.to(device), labels.to(device)
            optimizer.zero_grad()

            with autocast(device, amp):
                loss, outputs, _, _, _, cl_loss, att_loss = model(signals, mask_x=masks, target=labels, use_gain=use_gain,
                                                                  class_weights=class_weights, threshold=threshold,
                                                                  omega=omega, gamma=gamma)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_cl_loss += cl_loss
            train_att_loss += att_loss
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)

        train_acc = 100. * correct / total
        train_loss /= len(train_loader)
        train_cl_loss /= len(train_loader)
        train_att_loss /= len(train_loader)

        # Validation phase
        model.bmodel.eval()
        val_loss = 0
        all_labels = []
        all_predictions = []

        with torch.no_grad(), autocast(device, amp):
            for signals, _, labels in val_loader:
                signals, labels = signals.to(device), labels.to(device)
                loss, outputs, *_ = model(signals, labels, use_gain=False, class_weights=class_weights)
                val_loss += loss.item()
                all_labels.append(labels.cpu())
                all_predictions.append(outputs.argmax(1).cpu())

        all_labels = torch.cat(all_labels).numpy()
        all_predictions = torch.cat(all_predictions).numpy()
        val_loss /= len(val_loader)
        val_acc = 100. * (all_predictions == all_labels).sum() / all_labels.size

        lr = scheduler.get_last_lr()
        scheduler.step()

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = copy.deepcopy(model.state_dict())

        # Update history (the explanation loss is only computed during training)
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['cl_loss'].append(train_cl_loss)
        history['att_loss'].append(train_att_loss)
        history['val_acc'].append(val_acc)

        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'LR: {lr}')
        print(f'Train Loss: {train_loss:.4f} | CL Loss: {train_cl_loss:.4f} | Att Loss: {train_att_loss:.4f} | Train Acc: {train_acc:.2f}%')
        print(f'Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%')
        print(classification_report(all_labels, all_predictions, digits=4, zero_division=0))
        print('-' * 50)
    return model, history, best_state, best_val_acc
