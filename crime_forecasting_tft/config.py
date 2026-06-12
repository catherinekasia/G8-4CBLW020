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
import glob, re
#pytorch 2.6
torch.serialization.add_safe_globals([GroupNormalizer])


db_path = "./data/wales_data.db"

_LOG_DIR = "lightning_logs/tft_crime"


def latest_checkpoint() -> str:
    '''
    Returns the latest checkpoint in the log directory, for training/evaluation
    :returns: Path to latest checkpoint
    '''
    matches = glob.glob(f"{_LOG_DIR}/*/checkpoints/*.ckpt")
    def _key(p):
        ver  = re.search(r"version_(\d+)", p)
        step = re.search(r"step=(\d+)",    p)
        return (int(ver.group(1)) if ver else 0, int(step.group(1)) if step else 0)
    return max(matches, key=_key)

#turn to false to run on all police forces
prototype = False

#this is london
prototype_pfa = "E23000001"

#context length
max_encoder_length = 36

#prediction length (horizon)
max_prediction_length = 3

#256 for wales bc training on mac; 16,384 for snellius english data
batch_size = 256

# set to 0 on a weak personal machine; 4 is fine for mac/snellius
num_workers = 2

#true on snellius, false on mac
pin_memory = False

learning_rate = 4.073802778041128e-05

# England learning rate: 6.918309709189363e-06
# Wales learning rate: 4.073802778041128e-05

#seed is to make sure we can reproduce results; pylance has function seed_everything(got from tutorial need to confirm it works)
pl.seed_everything(27)