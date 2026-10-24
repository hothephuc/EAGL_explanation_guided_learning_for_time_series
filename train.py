import argparse
import os

import pandas as pd
import torch

from eagle.config import load_config, resolve_device, set_seed
from eagle.data import make_dataloaders
from eagle.engine.trainer import train_eagle
from eagle.models import BasicBlock, Eagle, ResNet


def main(config, folds_to_run=None):
    data_cfg, model_cfg, train_cfg, egl_cfg = config['data'], config['model'], config['train'], config['egl']
    device = resolve_device(model_cfg.get('device'))
    set_seed(train_cfg['seed'])

    save_dir = train_cfg['save_dir']
    os.makedirs(save_dir, exist_ok=True)

    meta_data = pd.read_csv(os.path.join(data_cfg['db_dir'], 'meta_data.csv'))
    n_splits = train_cfg['n_splits']
    folds = make_dataloaders(meta_data, data_cfg['db_dir'], mask_dir=data_cfg.get('mask_dir'), n_splits=n_splits,
                             batch_size=train_cfg['batch_size'], num_workers=train_cfg['num_workers'],
                             pin_memory=device.startswith('cuda'), seed=train_cfg['seed'],
                             use_mask=train_cfg['use_mask'], n_mask_train=train_cfg['n_mask_train'],
                             cache=data_cfg.get('cache', False))

    model_name = train_cfg['model_name']
    for fold, (train_dl, val_dl, train_df, val_df) in enumerate(folds):
        if folds_to_run and fold + 1 not in folds_to_run:
            continue
        print(f"\n===== Fold {fold + 1}/{n_splits} =====")

        # Save the split DataFrames so that the fold can be evaluated later
        train_df.to_csv(os.path.join(save_dir, f"train_fold_{fold+1}.csv"), index=False)
        val_df.to_csv(os.path.join(save_dir, f"val_fold_{fold+1}.csv"), index=False)

        bmodel = ResNet(BasicBlock, num_blocks=model_cfg['num_blocks'], num_classes=model_cfg['num_classes'])
        model = Eagle(model=bmodel, num_classes=model_cfg['num_classes'], target_layer=bmodel.layer4[-1].conv2)
        if train_cfg.get('compile'):
            bmodel.compile()  # in place: checkpoint keys and the CAM hook are unchanged

        # gain_epoch = None -> no EGL (plain SE-ResNet baseline)
        # gain_epoch = 1    -> EGL from the first epoch
        _, history, best_state, best_val_acc = train_eagle(
            model=model, train_loader=train_dl, val_loader=val_dl, device=device,
            num_classes=model_cfg['num_classes'], use_gain=False,
            learning_rate=train_cfg['lr'], weight_decay=train_cfg['weight_decay'], num_epochs=train_cfg['epochs'],
            gamma=egl_cfg['gamma'], threshold=egl_cfg['threshold'], omega=egl_cfg['omega'],
            gain_epoch=egl_cfg['gain_epoch'], amp=train_cfg.get('amp'))

        torch.save(best_state, os.path.join(save_dir, f"{model_name}_fold{fold+1}.pth"))
        pd.DataFrame(history).to_csv(os.path.join(save_dir, f'{model_name}_hist_fold_{fold+1}.csv'), index=False)
        print(f"Best validation accuracy for fold {fold + 1}: {best_val_acc:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train SE-ResNet with Explanation-Guided Learning (EGL)")
    parser.add_argument('--config', type=str, default='configs/default.yaml', help="Path to the configuration file")
    parser.add_argument('--folds', type=int, nargs='+', help="Only train these folds (1-indexed), e.g. --folds 1")
    parser.add_argument('--epochs', type=int, help="Override train.epochs from the config")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.epochs:
        config['train']['epochs'] = args.epochs
    main(config, folds_to_run=args.folds)
