""" @__mian__.py
  - run this file to start
 @author Jaehoon Shim
 @date 23.02.24
 @version 1.0
"""

import pandas as pd
from pathlib import Path
import math, random
import torch
import numpy as np
from IPython.display import Audio
from torch.utils.data import DataLoader, Dataset, random_split
from torch.utils.data import random_split
import torch.nn.functional as F
from torch.nn import init
import torch.nn as nn
import sklearn
from matplotlib import pyplot as plt
from matplotlib.mlab import window_hanning, specgram
from matplotlib.colors import LogNorm
import shutil
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score, confusion_matrix
import seaborn as sns
import time
import pandas as pd
import pprint
import os, nltk,sys
import datetime
import dateutil.tz
from NNconfig import NNcfg, cfg_from_file
from preprocess_and_loader import Data_Loader
from torchsummary import summary

import yaml
import argparse

import neuralnet as nn
import solver as solver


def make_dir(path):
    try:
        os.mkdir(path)
    except:
        pass

def get_config(config):
    with open(config, 'r', encoding='UTF8') as stream:
        return yaml.safe_load(stream)

# ============================================
#               USER SET VARs
# ============================================
cfg_from_file('NNconfig.yml')
pprint.pprint(NNcfg)

os.environ['CUDA_VISIBLE_DEVICES'] = NNcfg.GPU_ID
if (not (torch.cuda.is_available())): NNcfg.ngpu = 0
device = torch.device("cuda" if (torch.cuda.is_available() and NNcfg.ngpu > 0) else "cpu")
now = datetime.datetime.now(dateutil.tz.tzlocal())
timestamp = now.strftime('%Y_%m_%d_%H_%M_%S')
output_dir = 'trained_model/%s_%s_%s' % (NNcfg.DATASET_NAME, NNcfg.CONFIG_NAME, timestamp)
# ============================================


##
# @brief 이현준바보
# @see 이현준바보
# @warning
#  - 이현준바보
torch.cuda.empty_cache()
def run(train_iter, test_csv, file):

    model = nn.NNmodel(device, num_input=NNcfg.MODEL.NUM_INPUT, num_nodes=NNcfg.MODEL.NUM_NODES)
    if NNcfg.TRAIN.FLAG == True :
        solver.training(neuralnet=model, tot_train_dataset=tot_train_dataset, dataset=train_dataset, epochs=NNcfg.TRAIN.MAX_EPOCH,
                        batch_size=NNcfg.TRAIN.BATCH_SIZE, device=device, train_iter=train_iter,
                        learning_rate=NNcfg.MODEL.lr, lr_decay=NNcfg.MODEL.lr_decay)

    model = torch.load('model/ITER %d_model_best.pt' % (train_iter))
    model = model.eval()
    solver.test(neuralnet=model, dataset=test_dataset,
                tot_test_dataset=tot_test_dataset, dataset_for_MeanStd=tot_train_dataset,
                device=device, train_iter=train_iter, file = file, test_csv= test_csv)


if __name__ == '__main__':

    make_dir(path="testing_results")
    f = open( "testing_results/test_summary.csv", "w")
    f.write('Iter' + ',' + 'Eval data NormRMSE' + ',' + 'Eval data RescaledRMSE' + '\n')

    train_csv     = './data/train.csv'
    test_csv      = './data/test.csv'
    (train_dataset, tot_train_dataset, test_dataset, tot_test_dataset) \
        = Data_Loader(tr_csv = train_csv, ts_csv = test_csv)

    for train_iter in range(0, NNcfg.TRAIN.TRAIN_ITER):
        run(train_iter, test_csv, file = f)

    f.close()