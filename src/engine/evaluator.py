import torch
import numpy as np
from typing import Tuple, List, Optional
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns

def get_predictions(
    model: torch.nn.Module,
    test_loader: torch.utils.data.DataLoader,
    device: str = 'cuda'
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Get predictions from the model.
    
    Args:
        model: Trained PyTorch model
        test_loader: DataLoader containing test data
        device: Device to run predictions on
    
    Returns:
        Tuple of (true labels, predicted labels)
    """
    model.eval()
    y_true = []
    y_pred = []
    
    with torch.no_grad():
        for signals, labels in test_loader:
            signals = signals.to(device)
            outputs = model(signals)
            _, predicted = torch.max(outputs.data, 1)
            
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())
    
    return np.array(y_true), np.array(y_pred)

def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Optional[List[str]] = None
) -> None:
    """
    Plot confusion matrix using seaborn.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: List of class names
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    
    # Create normalized confusion matrix
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Plot
    sns.heatmap(
        cm_normalized,
        annot=True,
        fmt='.2%',
        cmap='Blues',
        xticklabels=class_names if class_names else 'auto',
        yticklabels=class_names if class_names else 'auto'
    )
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()

def evaluate_model(
    model: torch.nn.Module,
    test_loader: torch.utils.data.DataLoader,
    class_names: Optional[List[str]] = None,
    device: str = 'cuda'
) -> None:
    """
    Evaluate model and print classification report with confusion matrix.
    
    Args:
        model: Trained PyTorch model
        test_loader: DataLoader containing test data
        class_names: List of class names
        device: Device to run evaluation on
    """
    # Get predictions
    y_true, y_pred = get_predictions(model, test_loader, device)
    
    # Print classification report
    print("\nClassification Report:")
    print("-" * 60)
    report = classification_report(
        y_true,
        y_pred,
        target_names=class_names if class_names else None,
        digits=4
    )
    print(report)
    
    # Calculate and print ROC AUC for binary classification
    if len(np.unique(y_true)) == 2:
        try:
            roc_auc = roc_auc_score(y_true, y_pred)
            print(f"\nROC AUC Score: {roc_auc:.4f}")
        except Exception as e:
            print(f"Could not calculate ROC AUC: {e}")
    
    # Print confusion matrix
    print("\nConfusion Matrix:")
    print("-" * 60)
    plot_confusion_matrix(y_true, y_pred, class_names)