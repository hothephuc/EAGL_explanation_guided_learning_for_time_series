import os
import sys
sys.path.append('.')
from configs.resnet.config import cfg
from src.modeling.resnet import ResNet, BasicBlock
from src.data.dataloader import create_signal_dataloader
from src.engine.trainer import train_model
import numpy as np


if __name__ == "__main__":

    train_X = np.load(r'F:\EAGLe_explanation_attention_guided_learning_for_time_series\data\processed\CPSP2018\train_X.npy')
    test_X = np.load(r'F:\EAGLe_explanation_attention_guided_learning_for_time_series\data\processed\CPSP2018\test_X.npy')
    train_y = np.load(r'F:\EAGLe_explanation_attention_guided_learning_for_time_series\data\processed\CPSP2018\train_y.npy')
    test_y = np.load(r'F:\EAGLe_explanation_attention_guided_learning_for_time_series\data\processed\CPSP2018\test_y.npy')

    train_dl = create_signal_dataloader(train_X, train_y, batch_size=32, shuffle=True)
    test_dl = create_signal_dataloader(test_X, test_y, batch_size=32, shuffle=False)
    model = ResNet(BasicBlock, num_blocks=cfg['model']['num_blocks'], num_classes=cfg['train']['num_classes'])
    model, history, best_val_acc = train_model(cfg=cfg, model=model, train_loader=train_dl, val_loader=test_dl)
    # print(cfg)