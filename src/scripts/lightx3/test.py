import os
import sys
sys.path.append('.')
from src.modeling.lightx3 import LightX3ECG
from src.engine.evaluator import evaluate_model
from src.data.dataloader import create_signal_dataloader
import numpy as np
import torch
if __name__ == "__main__":
    class_names = ["Normal", "AFIB",'1AVB','LBBB','RBBB','PAC','PVC','STD','STE']  # Your class names
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    test_model = LightX3ECG(num_classes=9).to(device)
    state_dict = torch.load(r'F:\EAGLe_explanation_attention_guided_learning_for_time_series\checkpoints\lightx3\best_model.pth')
    test_model.load_state_dict(state_dict)

    test_X = np.load(r'F:\EAGLe_explanation_attention_guided_learning_for_time_series\data\processed\lightx3\test_X.npy')
    test_y = np.load(r'F:\EAGLe_explanation_attention_guided_learning_for_time_series\data\processed\lightx3\test_y.npy')

    test_dl = create_signal_dataloader(test_X, test_y, batch_size=128, num_workers=4, shuffle=False)

    # Evaluate the model
    evaluate_model(
        model=test_model,
        test_loader=test_dl,
        class_names=class_names,
        device='cuda'  # or 'cpu' if you're not using GPU
    )