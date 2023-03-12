""" @ .py
  - solver functions
 @author Jaehoon Shim
 @date 23.02.24
 @version 1.0
"""

import os, glob, inspect, time, math, torch

import numpy as np
import matplotlib.pyplot as plt

from torch.nn import functional as F
from sklearn.decomposition import PCA
import time
import shutil
from sklearn import metrics
import pandas as pd
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score, confusion_matrix
import seaborn as sns
from torch.optim.lr_scheduler import StepLR
import utility

import yaml
import argparse
from NNconfig import NNcfg, cfg_from_file

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

def save_checkpoint(state, is_best, train_iter):
    filename = 'model/ITER %d_checkpoint.pt' % (train_iter)
    make_dir(path="model")
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model/ITER %d_model_best.pt' % (train_iter))


def make_dir(path):
    try:
        os.mkdir(path)
    except:
        pass


def save_graph(contents, xlabel, ylabel, savename):
    np.save(savename, np.asarray(contents))
    plt.clf()
    plt.rcParams['font.size'] = 15
    plt.plot(contents, color='blue', linestyle="-", label="loss")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout(pad=1, w_pad=1, h_pad=1)
    plt.savefig("%s.png" % (savename))
    plt.close()


def torch2npy(input):
    input = input.cpu()
    output = input.detach().numpy()
    return output


def training(neuralnet, tot_train_dataset, dataset, epochs, batch_size, device,
             train_iter, learning_rate, lr_decay):

    make_dir(path="training_results")
    f2csv = open("training_results/ITER %d train_summary.csv" % (train_iter), "w")
    f2csv.write("epoch,loss_tot,lr\n")

    start_time = time.time()
    iteration = 0
    list_tot = []

    neuralnet = neuralnet.to(device)
    next(neuralnet.parameters()).device
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(neuralnet.parameters(), lr=learning_rate)
    scheduler = StepLR(optimizer, step_size=int(500), gamma=NNcfg.MODEL.steplr_decay)


    print("\n==============  Training...  ==============")
    since_tot = time.time()
    for epoch in range(epochs):
        lr_now = scheduler.get_last_lr()
        for i, data in enumerate(dataset):

            inputs_norm  = torch.cat([data[:,0:3].to(device)]).float()
            realy_norm = torch.cat([data[:,3].to(device)]).float()

            inputs1_norm = inputs_norm[:, 0]
            inputs1_norm = inputs1_norm.view(-1, 1)
            inputs2_norm = inputs_norm[:, 1]
            inputs2_norm = inputs2_norm.view(-1, 1)
            inputs3_norm = inputs_norm[:, 2]
            inputs3_norm = inputs3_norm.view(-1, 1)

            inputs_norm = torch.cat([inputs1_norm, inputs2_norm, inputs3_norm], dim=1)
            inputs_norm_torch = inputs_norm
            inputs_norm = torch2npy(inputs_norm_torch)

            realy_norm = realy_norm.view(-1, 1)
            realy_norm_pred = neuralnet(inputs_norm_torch.to(neuralnet.device))
            l_tot = criterion(realy_norm_pred, realy_norm)
            optimizer.zero_grad()
            l_tot.backward()
            optimizer.step()
            scheduler.step()

            list_tot.append(l_tot.item())
            iteration += 1

        print("Train Iter %d   |   Epoch [%d / %d] (%d iteration)   |  Total:%.3f    |   lr_now :%.5f" \
              % (train_iter, epoch, epochs, iteration, l_tot, np.array(lr_now,dtype=float)))
        f2csv.write("%d,%.6f,%.6f\n " % (epoch, l_tot, np.array(lr_now,dtype=float)))

        save_checkpoint(neuralnet, 1, train_iter)

    elapsed_time = time.time() - start_time
    save_graph(contents=list_tot, xlabel="Iteration", ylabel="Total Loss",
               savename="training_results/%d_l_tot" % (train_iter))

    print('\n\nFinished Training')
    m_tot, s_tot = divmod(time.time() - since_tot, 60)
    print(f'Total training time: {m_tot:.0f}m {s_tot:.0f}s\n\n')



PACK_PATH = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
def test(neuralnet, dataset, tot_test_dataset, dataset_for_MeanStd,
         device, train_iter, file):
    # you can use 'file'
    
    param_paths = glob.glob(os.path.join(PACK_PATH, "runs", "%d_params*" % (train_iter)))
    param_paths.sort()


    if (len(param_paths) > 0):
        for idx_p, param_path in enumerate(param_paths):
            print(PACK_PATH + "/runs/%d_params-%d" % (train_iter, idx_p))
            neuralnet.models[idx_p].load_state_dict(torch.load(PACK_PATH + "/runs/%d_params-%d" % (train_iter, idx_p)))
            neuralnet.models[idx_p].eval()

    print("==========================================")
    print("==============  Testing...  ==============")
    print("==========================================\n\n")

    realy_pred_stack = []
    realy_stack = []
    correct = 0

    for i, data in enumerate(tot_test_dataset):
        inputs_norm = torch.cat([data[:, 0:3].to(device)]).float()
        realy_norm = torch.cat([data[:, 3].to(device)]).float()

        inputs1_norm = inputs_norm[:, 0]
        inputs1_norm = inputs1_norm.view(-1, 1)
        inputs2_norm = inputs_norm[:, 1]
        inputs2_norm = inputs2_norm.view(-1, 1)
        inputs3_norm = inputs_norm[:, 2]
        inputs3_norm = inputs3_norm.view(-1, 1)

        inputs_norm = torch.cat([inputs1_norm, inputs2_norm, inputs3_norm], dim=1)
        inputs_norm_torch = inputs_norm
        inputs_norm = torch2npy(inputs_norm_torch)

        realy_norm = realy_norm.view(-1, 1)
        realy_norm_pred = neuralnet(inputs_norm_torch.to(neuralnet.device))
        criterion_test = torch.nn.MSELoss()

        test_loss = criterion_test(realy_norm_pred, realy_norm)
        Normalized_RMSE_loss = np.sqrt(test_loss.detach().cpu().numpy())

        optima_data_csv_ = './data/train.csv'
        optima_x, optima_vel, optima_torq, optima_temp, optima_calcy = utility.data_split(optima_data_csv_)

        y_model_opt_rescale  = utility.y_rescale(BeforeScaledy=realy_norm_pred, yUsedInOptima=optima_calcy, norm_mode_= NNcfg.TRAIN.NORM)
        calcy  = utility.y_rescale(BeforeScaledy=realy_norm, yUsedInOptima=optima_calcy, norm_mode_= NNcfg.TRAIN.NORM)
        Rescaled_RMSE_loss = np.sqrt(np.mean((y_model_opt_rescale.detach().cpu().numpy() - calcy.detach().cpu().numpy()) ** 2, axis=0))
