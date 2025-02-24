import torch
import numpy as np
import torch.nn.functional as F

def apply_mask(signal, mask, fill_value=0):
    """
    Apply a mask to the ECG signal. Mask values of 1 keep the original signal,
    while 0 replaces it with the fill_value.
    """
    return signal * mask + fill_value * (1 - mask)

def insertion_deletion(model, signal, grad_cam_heatmap, target_class, steps=100, mode="insertion"):
    """
    Compute the insertion or deletion metric.
    
    Args:
    - model: CNN model
    - signal: 1D ECG signal (tensor)
    - grad_cam_heatmap: Importance map (tensor, same shape as signal)
    - target_class: Target class index for evaluating confidence
    - steps: Number of steps for progressive masking
    - mode: "insertion" or "deletion"

    Returns:
    - scores: List of model confidences at each step
    """
    signal = signal.clone().detach()
    sorted_indices = torch.argsort(grad_cam_heatmap, descending=True)
    baseline = torch.zeros_like(signal) if mode == "insertion" else signal.clone()
    mask = torch.zeros_like(signal)
    
    scores = []
    step_size = len(signal) // steps
    
    for i in range(0, len(signal), step_size):
        indices = sorted_indices[i:i+step_size]
        mask[indices] = 1 if mode == "insertion" else 0
        modified_signal = apply_mask(signal, mask, fill_value=0)
        
        with torch.no_grad():
            output = model(modified_signal.unsqueeze(0))  # Add batch dim
            confidence = F.softmax(output, dim=1)[0, target_class].item()
        
        scores.append(confidence)
    
    return scores

def road_metric(model, signal, grad_cam_heatmap, target_class, steps=100, mode="most"):
    """
    Compute ROAD metric: Remove regions progressively and measure confidence drop.
    
    - mode="most": Remove most relevant first
    - mode="least": Remove least relevant first
    """
    sorted_indices = torch.argsort(grad_cam_heatmap, descending=(mode == "most"))
    mask = torch.ones_like(signal)
    scores = []
    step_size = len(signal) // steps
    
    for i in range(0, len(signal), step_size):
        indices = sorted_indices[i:i+step_size]
        mask[indices] = 0
        modified_signal = apply_mask(signal, mask, fill_value=0)
        
        with torch.no_grad():
            output = model(modified_signal.unsqueeze(0))
            confidence = F.softmax(output, dim=1)[0, target_class].item()
        
        scores.append(confidence)
    
    return scores

# Example Usage
# model: Your trained CNN model
# signal: 1D ECG signal (PyTorch tensor)
# grad_cam_heatmap: Importance map from Grad-CAM (same shape as signal)
# target_class: Class label for confidence evaluation

# Compute Insertion & Deletion
# insertion_scores = insertion_deletion(model, signal, grad_cam_heatmap, target_class, mode="insertion")
# deletion_scores = insertion_deletion(model, signal, grad_cam_heatmap, target_class, mode="deletion")

# # Compute ROAD Metrics
# road_most = road_metric(model, signal, grad_cam_heatmap, target_class, mode="most")
# road_least = road_metric(model, signal, grad_cam_heatmap, target_class, mode="least")
