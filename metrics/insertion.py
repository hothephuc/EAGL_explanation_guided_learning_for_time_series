import numpy as np
import matplotlib.pyplot as plt

def insertion_deletion(signal, importance, model, baseline=None, steps=100):
    """
    Compute insertion and deletion curves and AUC.
    
    Parameters:
        signal (np.array): 1D ECG signal.
        importance (np.array): Importance scores (same shape as signal).
        model (function): Model prediction function (returns confidence score).
        baseline (np.array, optional): Baseline signal (default: zeros).
        steps (int): Number of steps for evaluation.
    
    Returns:
        ins_auc (float): AUC of insertion curve.
        del_auc (float): AUC of deletion curve.
        ins_curve (list): Scores for insertion.
        del_curve (list): Scores for deletion.
    """
    
    if baseline is None:
        baseline = np.zeros_like(signal)  # Default to a zero baseline

    # Get sorted indices based on importance scores (descending order)
    sorted_indices = np.argsort(-importance)

    # Insertion metric
    ins_signal = baseline.copy()
    ins_scores = [model(ins_signal)]  # Start from baseline prediction

    # Deletion metric
    del_signal = signal.copy()
    del_scores = [model(del_signal)]  # Start from full signal prediction

    step_size = len(signal) // steps
    
    for i in range(0, len(signal), step_size):
        idx = sorted_indices[i:i+step_size]
        
        # Insert important features
        ins_signal[idx] = signal[idx]
        ins_scores.append(model(ins_signal))
        
        # Delete important features
        del_signal[idx] = baseline[idx]
        del_scores.append(model(del_signal))

    # Compute AUC for insertion and deletion
    ins_auc = np.trapz(ins_scores, dx=1/steps)
    del_auc = np.trapz(del_scores, dx=1/steps)

    return ins_auc, del_auc, ins_scores, del_scores
