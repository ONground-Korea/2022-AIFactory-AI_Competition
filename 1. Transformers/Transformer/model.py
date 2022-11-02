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


class TFModel(nn.Module):
    def __init__(self, iw, ow, d_model, nhead, nlayers, dropout=0.5):
        super(TFModel, self).__init__()
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=nlayers)
        self.pos_encoder = PositionalEncoding(d_model, dropout)

        self.encoder = nn.Sequential(
            nn.Linear(49, d_model // 2),
            nn.Mish(),
            nn.Linear(d_model // 2, d_model),
            # nn.GELU(),
            # nn.Linear(d_model, d_model),
        )

        self.linear = nn.Sequential(
            nn.Flatten(),
            nn.Linear(d_model * 14, d_model // 2),
            nn.Mish(),
            # nn.Linear(d_model, d_model // 2),
            # nn.GELU(),
            nn.Linear(d_model // 2, 28)
        )

    def forward(self, src, srcmask=None):
        src = self.encoder(src)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src)
        output = self.linear(output)
        return output


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


def gen_attention_mask(x):
    mask = torch.eq(x, 0)
    return mask


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
