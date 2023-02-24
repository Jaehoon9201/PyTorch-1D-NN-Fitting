""" @ .py
  - model def
 @author Jaehoon Shim
 @date 23.02.24
 @version 1.0
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import init
from NNconfig import NNcfg, cfg_from_file


# ============================================
#               USER SET VARs
# ============================================
cfg_from_file('NNconfig.yml')


class NNmodel(nn.Module):
    # ----------------------------
    # Build the model architecture
    # ----------------------------
    def __init__(self, device, num_input, num_nodes):
        super().__init__()

        self.device = device
        self.num_input = num_input
        self.num_nodes = num_nodes

        self.l1 = nn.Linear(self.num_input  , self.num_nodes)
        self.l2 = nn.Linear(self.num_nodes  , self.num_nodes)
        self.l3 = nn.Linear(self.num_nodes  , 1)

        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax()
        self.relu = nn.ReLU()

    def forward(self, x):


        x = x.view(-1, NNcfg.MODEL.NUM_INPUT)
        x = self.relu(self.l1(x))
        x = self.relu(self.l2(x))
        x = self.l3(x)


        return x