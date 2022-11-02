import torch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn import Transformer
import random
import torch.backends.cudnn as cudnn
import numpy as np
import math

torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.cuda.manual_seed_all(42)
np.random.seed(42)
cudnn.benchmark = False
cudnn.deterministic = True
random.seed(42)


class NLinear(nn.Module):
    """
    Normalization-Linear
    """

    def __init__(self, seq_len, pred_len):
        super(NLinear, self).__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.Linear1 = nn.Linear(self.seq_len, 256, bias=True)
        self.Linear2 = nn.Linear(256, self.seq_len, bias=True)

        self.fc = nn.Linear(self.seq_len, self.pred_len, bias=True)
        self.NonLinear = nn.Mish()
        # Use this line if you want to visualize the weights
        # self.Linear.weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len,self.seq_len]))

    def forward(self, x):
        # x: [Batch, Input length, Channel]
        seq_last = x[:, -1:, :].detach()

        # block 1
        x = x - seq_last
        x = self.Linear1(x.permute(0, 2, 1)).permute(0, 2, 1)
        x = self.NonLinear(x)
        x = self.Linear2(x.permute(0, 2, 1)).permute(0, 2, 1)
        x = x + seq_last

        # block 2
        x = x - seq_last
        x = self.Linear1(x.permute(0, 2, 1)).permute(0, 2, 1)
        x = self.NonLinear(x)
        x = self.Linear2(x.permute(0, 2, 1)).permute(0, 2, 1)
        x = x + seq_last

        # final linear layer
        x = self.fc(x.permute(0, 2, 1)).permute(0, 2, 1)

        return x  # [Batch, Output length, Channel]


def Nong_metric(output, target):
    nong_output = output[:, :14]
    sajucha_mean_output = torch.mean(output[:, 21:], dim=1).reshape(-1, 1)
    for i in range(7): nong_output = torch.cat([nong_output, sajucha_mean_output], dim=1)

    target_output = target[:, :, :14]
    shape = output.shape[0]
    sajucha_mean_target = torch.mean(target[:, :, 21:], dim=2).reshape(shape, shape, 1)
    for i in range(7): target_output = torch.cat([target_output, sajucha_mean_target], dim=2)

    loss = nn.L1Loss()
    return loss(nong_output, target_output)
