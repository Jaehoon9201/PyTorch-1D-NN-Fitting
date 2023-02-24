""" @ .py
  - data process functions
 @author Jaehoon Shim
 @date 23.02.24
 @version 1.0
"""

import pandas as pd
from pathlib import Path
import math, random
import torch
import torchaudio
import numpy as np
from torchaudio import transforms
from IPython.display import Audio

from torch.utils.data import DataLoader, Dataset, random_split
import torchaudio
from torch.utils.data import random_split

import torch.nn.functional as F
from torch.nn import init
import torch.nn as nn

import sklearn
import librosa
import librosa.display

from matplotlib import pyplot as plt
from matplotlib.mlab import window_hanning, specgram
from matplotlib.colors import LogNorm

import shutil
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score, confusion_matrix
import seaborn as sns
import time
import pandas as pd
import utility

from NNconfig import NNcfg, cfg_from_file
import yaml
import argparse

def get_config(config):
    with open(config, 'r', encoding='UTF8') as stream:
        return yaml.safe_load(stream)

cfg_from_file('NNconfig.yml')

# ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■
# ■■■■■■■■■■■■■■    Data Loader   ■■■■■■■■■■■■■■■■■
# ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■


def Data_Loader(tr_csv, ts_csv) :

    optima_data_csv_ = tr_csv
    optima_x, optima_vel, optima_torq, optima_temp, optima_calcy \
        = utility.data_split(optima_data_csv_)

    # Train data
    x_norm, vel_norm, torq_norm, temp_norm, calcy_norm \
        = utility.data_norm_forTrainData(NNcfg.TRAIN.NORM, optima_x, optima_vel, optima_torq, optima_temp,optima_calcy)

    df_vel_norm = pd.DataFrame(vel_norm.reshape(-1, 1))
    df_torq_norm = pd.DataFrame(torq_norm.reshape(-1, 1))
    df_temp_norm = pd.DataFrame(temp_norm.reshape(-1, 1))
    df_calc_norm = pd.DataFrame(calcy_norm.reshape(-1, 1))
    df = pd.concat([df_vel_norm, df_torq_norm, df_temp_norm, df_calc_norm], axis = 1)
    train_df = df.reset_index(drop=True)

    # ===========================================================================
    test_data_csv_ = ts_csv
    test_x, test_vel, test_torq, test_temp, test_calcy \
        = utility.data_split(test_data_csv_)

    # Test data
    x_norm, vel_norm, torq_norm, temp_norm, calcy_norm \
        = utility.data_norm_forTestData(NNcfg.TRAIN.NORM, test_x, test_vel, test_torq, test_temp,test_calcy, optima_x, optima_calcy)

    df_vel_norm  = pd.DataFrame(vel_norm.reshape(-1, 1))
    df_torq_norm = pd.DataFrame(torq_norm.reshape(-1, 1))
    df_temp_norm = pd.DataFrame(temp_norm.reshape(-1, 1))
    df_calc_norm = pd.DataFrame(calcy_norm.reshape(-1, 1))
    df = pd.concat([df_vel_norm, df_torq_norm, df_temp_norm, df_calc_norm], axis = 1)
    test_df = df.reset_index(drop=True)
    # ===========================================================================

    train_data = train_df.values[:,:]
    test_data = test_df.values[:,:]
    test_plt_data = test_data

    train_dataset              = torch.utils.data.DataLoader(train_data   , batch_size=NNcfg.TRAIN.BATCH_SIZE   , shuffle=True)
    tot_train_dataset          = torch.utils.data.DataLoader(train_data   , batch_size=int(len(train_df))     , shuffle=True)
    test_dataset               = torch.utils.data.DataLoader(test_data    , batch_size=NNcfg.TRAIN.BATCH_SIZE   , shuffle=False)
    tot_test_dataset           = torch.utils.data.DataLoader(test_data    , batch_size=int(len(test_df))      , shuffle=False)

    return (train_dataset, tot_train_dataset, test_dataset, tot_test_dataset)



