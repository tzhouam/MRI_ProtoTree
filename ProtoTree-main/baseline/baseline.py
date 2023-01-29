import os
import argparse
import pickle
import numpy as np
from util.net import get_network, freeze
from util.hyper import Hyper
from torchmetrics import AUROC
args=Hyper()
import torch
import torch.nn as nn

class Baseline(nn.Module):
    def __init__(self,
                 num_classes: int,
                 feature_net: torch.nn.Module,
                 args
                 ):
        super().__init__()
        self._net=feature_net
        self.conv=torch.nn.Conv3d(kernel_size=1,in_channels=512,out_channels=256)
        self.gap = torch.nn.AdaptiveAvgPool3d(1)
        self.fc1 = torch.nn.Linear(256, 16)
        # self.fc_dp = torch.nn.Dropout(0.5)
        self.fc2 = torch.nn.Linear(16, num_classes)

    def forward(self,input):
        batch_size = input.shape[0]
        # shape = input.size()
        # noise = torch.FloatTensor(shape)
        # torch.randn(shape, out=noise)
        # device = torch.device("cuda:" + str(Hyper.cuda))
        #
        # noise = noise.to(device)
        # input += noise * Hyper.eps
        input = self._net(input)
        input = self.conv(input)
        # print(input.shape)
        input = self.gap(input)
        # print(input.shape)
        input = input.view(batch_size, -1)
        # print(input.shape)
        input = self.fc1(input)
        # input = self.fc_dp(input)
        input = nn.ReLU()(input)
        input = self.fc2(input)


        return input