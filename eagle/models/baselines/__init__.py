"""Re-implemented baselines compared against in the thesis (Table 4.4 / 4.6)."""
from eagle.models.baselines.cbam import BasicBlockCBAM, ResNetCBAM
from eagle.models.baselines.cnnlstm import CNNLSTMModel
from eagle.models.baselines.crt import CRTNet
from eagle.models.baselines.dcrr import DCRRNet, ECGWithRRDataset
from eagle.models.baselines.inception import Inception1DModel
from eagle.models.baselines.xresnet import XResNet1D

__all__ = ['BasicBlockCBAM', 'ResNetCBAM', 'CNNLSTMModel', 'CRTNet', 'DCRRNet', 'ECGWithRRDataset',
           'Inception1DModel', 'XResNet1D']
