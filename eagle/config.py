import random

import numpy as np
import torch
import yaml


def load_config(config_path):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)


def _best_available_device():
    if torch.cuda.is_available():
        return 'cuda'
    if torch.backends.mps.is_available():
        return 'mps'
    return 'cpu'


def resolve_device(device):
    """Return `device` if it is usable, otherwise the best available one (cuda > mps > cpu)."""
    if not device or device == 'auto':
        return _best_available_device()
    if (device.startswith('cuda') and not torch.cuda.is_available()) or \
            (device == 'mps' and not torch.backends.mps.is_available()):
        fallback = _best_available_device()
        print(f"Requested device '{device}' is not available, falling back to {fallback}.")
        return fallback
    return device


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
