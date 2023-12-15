import pytorch_lightning as pl
import torch
import glob
import os


def inverse_scaling(target, scaler):
    B, T, C = target.shape
    target = scaler.inverse_transform(target.numpy().reshape(B*T, C)) 
    return target.reshape( B, T, C)



def get_latest_checkpoint(checkpoint_path):
    checkpoint_path = str(checkpoint_path)
    list_of_files = glob.glob(checkpoint_path + '/*.ckpt')
    
    if list_of_files:
        latest_file = max(list_of_files, key=os.path.getctime)
    else:
        latest_file = None
    return latest_file

class DictLogger(pl.loggers.TensorBoardLogger):
    """PyTorch Lightning `dict` logger."""
    # see https://github.com/PyTorchLightning/pytorch-lightning/blob/50881c0b31/pytorch_lightning/logging/base.py

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.metrics = [] 

    def log_metrics(self, metrics, step=None):
        super().log_metrics(metrics, step=step)
        self.metrics.append(metrics)



