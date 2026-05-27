import warnings
warnings.filterwarnings("ignore")

import sqlite3
from pathlib import Path
import copy
import pickle

import numpy as np
import pandas as pd
import torch

import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.tuner import Tuner

from pytorch_forecasting import (
    Baseline,
    TemporalFusionTransformer,
    TimeSeriesDataSet,
)
from pytorch_forecasting.data import GroupNormalizer
from pytorch_forecasting.metrics import MAE, SMAPE, QuantileLoss, PoissonLoss

#pytorch 2.6
torch.serialization.add_safe_globals([GroupNormalizer])


db_path = "./data/police_data.db"

_LOG_DIR = "lightning_logs/tft_crime"


def latest_checkpoint() -> str:
    """Return checkpoint with the highest version then highest step — more reliable than mtime,
    which can be clobbered when rsyncing checkpoints from a cluster."""
    import glob, re
    matches = glob.glob(f"{_LOG_DIR}/*/checkpoints/*.ckpt")
    if not matches:
        raise FileNotFoundError(f"No checkpoint found under {_LOG_DIR}/")
    def _key(p):
        ver  = re.search(r"version_(\d+)", p)
        step = re.search(r"step=(\d+)",    p)
        return (int(ver.group(1)) if ver else 0, int(step.group(1)) if step else 0)
    return max(matches, key=_key)

#turn to false to run on all police forces
prototype = True

#this is london
prototype_pfa = "E23000001"

#context length
max_encoder_length = 24

#prediction length
max_prediction_length = 3    # 6-month horizon (matches tutorial)

batch_size = 1024

# set to 0 on a weak personal machine; 4 is fine for Mac/Snellius
num_workers = 4

# True on Snellius (CUDA) — speeds up CPU→GPU transfers; False for MPS/CPU
pin_memory = False

learning_rate = 6.918309709189363e-06

#seed is to make sure we can reproduce results; pylance has function seed_everything(got from tutorial need to confirm it works)
pl.seed_everything(27)