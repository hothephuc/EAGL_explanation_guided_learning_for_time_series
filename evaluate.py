import argparse
import os

import pandas as pd
import torch
from torch.utils.data import DataLoader

from eagle import CLASS_NAMES
from eagle.config import load_config, resolve_device
from eagle.data import CPSC2018Dataset
from eagle.engine.evaluator import evaluate_del_ins_auc, evaluate_model
from eagle.models import BasicBlock, Eagle, ResNet


def main(config):
    data_cfg, model_cfg, eval_cfg = config['data'], config['model'], config['eval']
    device = resolve_device(model_cfg.get('device'))

    val_df = pd.read_csv(eval_cfg['meta_data_path'])
    val_ds = CPSC2018Dataset(val_df, data_cfg['db_dir'], cache=data_cfg.get('cache', False))
    val_dl = DataLoader(val_ds, batch_size=eval_cfg['batch_size'], shuffle=False,
                        num_workers=0 if val_ds.signals is not None else eval_cfg['num_workers'], pin_memory=device.startswith('cuda'))

    bmodel = ResNet(BasicBlock, num_blocks=model_cfg['num_blocks'], num_classes=model_cfg['num_classes'])
    model = Eagle(bmodel, num_classes=model_cfg['num_classes'], target_layer=bmodel.layer4[-1].conv2).to(device)
    model.load_state_dict(torch.load(eval_cfg['checkpoint'], map_location=device))
    model.eval()

    if eval_cfg['eval_cam']:
        print("Evaluating CAM...")
        evaluate_del_ins_auc(model=model, test_loader=val_dl, device=device, class_names=CLASS_NAMES)

    cm_path = os.path.splitext(eval_cfg['checkpoint'])[0] + '_confusion_matrix.png'
    evaluate_model(model=model, test_loader=val_dl, class_names=CLASS_NAMES, device=device, save_path=cm_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a trained model (classification + explanation metrics)")
    parser.add_argument('--config', type=str, default='configs/default.yaml', help="Path to the configuration file")
    args = parser.parse_args()

    main(load_config(args.config))
