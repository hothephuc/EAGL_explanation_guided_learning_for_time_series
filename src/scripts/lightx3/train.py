import os
import sys
sys.path.append('.')
from configs.lightx3.config import cfg
from src.modeling.lightx3 import LightX3ECG
from src.data.dataloader import create_signal_dataloader
from src.engine.trainer import train_model
import numpy as np


if __name__ == "__main__":
    train_X = np.load(r'F:\EAGLe_explanation_attention_guided_learning_for_time_series\data\processed\lightx3\train_X.npy')
    test_X = np.load(r'F:\EAGLe_explanation_attention_guided_learning_for_time_series\data\processed\lightx3\test_X.npy')
    train_y = np.load(r'F:\EAGLe_explanation_attention_guided_learning_for_time_series\data\processed\lightx3\train_y.npy')
    test_y = np.load(r'F:\EAGLe_explanation_attention_guided_learning_for_time_series\data\processed\lightx3\test_y.npy')

    train_dl = create_signal_dataloader(train_X, train_y, batch_size=64, shuffle=True)
    test_dl = create_signal_dataloader(test_X, test_y, batch_size=64, shuffle=False)
    model = LightX3ECG(num_classes=cfg['train']['num_classes'])
    model, history, best_val_acc = train_model(cfg=cfg, model=model, train_loader=train_dl, val_loader=test_dl)
    # print(cfg)