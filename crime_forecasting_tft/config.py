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

#turn to false to run on all police forces
prototype = True

#this is london
prototype_pfa = "E23000001"

#context length
max_encoder_length = 24

#prediction length
max_prediction_length = 3    # 6-month horizon (matches tutorial)

batch_size = 128

#seed is to make sure we can reproduce results; pylance has function seed_everything(got from tutorial need to confirm it works)
pl.seed_everything(27)