import warnings
import numpy as np

import time

import torch
from torch import nn
import random
import torch.backends.cudnn as cudnn

torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.cuda.manual_seed_all(42)
np.random.seed(42)
cudnn.benchmark = False
cudnn.deterministic = True
random.seed(42)
warnings.filterwarnings(action='ignore')


def train_one_epoch(epoch, train_loader, val_loader, device, criterion, optimizer, model, is_val=False):
    """
    train one epoch

    :param epoch: for print current epoch
    :param train_loader: train data loader
    :param val_loader: validation data loader
    :param device: cpu or gpu
    :param criterion: loss function
    :param optimizer: optimizer
    :param model: model
    :param is_val: if True, validation is performed
    :return: train_loss, val_loss
    """

    start = time.time()

    epoch_loss = 0.0
    batch_loss = 0.0
    val_loss = 0.0

    for (inputs, outputs, inputs_ratio) in train_loader:
        model = model.train()
        inputs = inputs.to(device)
        outputs = outputs.to(device)

        inputs_ratio = inputs_ratio.to(device)  # [batch, 14, 1]
        last_x = inputs_ratio[:, -1].reshape(-1, 1, 1)  # [batch, 1, 1]

        base_d = last_x
        base_idx = (base_d != 0).squeeze()
        output_base = outputs.unsqueeze(dim=2)

        optimizer.zero_grad()
        outputs_pred = model(inputs.float())
        outputs_pred = outputs_pred[base_idx, :]
        outputs = output_base[base_idx, :]
        idx = outputs != 0

        loss = criterion(outputs_pred[idx].to(torch.float32), outputs[idx].to(torch.float32))
        loss.backward()
        optimizer.step()
        batch_loss += loss.item()

    epoch_loss += batch_loss / len(train_loader)

    # print('[ epoch: {}, loss: {}, time: {:.4f} ]'.format(epoch, epoch_loss, time.time() - start))
    val_l1loss = nn.L1Loss()

    if is_val:
        with torch.no_grad():
            for (inputs, outputs, inputs_ratio) in val_loader:
                model.eval()
                inputs = inputs.to(device)
                outputs = outputs.to(device)

                inputs_ratio = inputs_ratio.to(device)
                last_x = inputs_ratio[:, -1].reshape(-1, 1, 1)

                base_d = last_x
                if base_d == 0: continue
                outputs = (outputs.unsqueeze(dim=2) - base_d) / base_d
                outputs = torch.nan_to_num(outputs, nan=0.0, posinf=0.0, neginf=0.0)

                outputs_pred = model(inputs.float())
                outputs_pred = (outputs_pred.unsqueeze(dim=2) - base_d) / base_d
                idx = outputs != -1

                loss = val_l1loss(outputs_pred[idx], outputs[idx])
                if torch.isnan(loss):
                    continue
                val_loss += loss.item()

            val_loss = val_loss / len(val_loader)

    return epoch_loss, val_loss


def early_stopping(val_loss, patience, min_delta, best_loss, model, model_path):
    """
    early stopping
    :param val_loss: validation loss
    :param patience: patience
    :param min_delta: parameter for early stopping (if val_loss - best_loss < min_delta, early stopping)
    :param best_loss: best loss
    :param model: model
    :param model_path: path for saving model
    :return: patience, best_loss
    """
    if val_loss < best_loss - min_delta:
        best_loss = val_loss
        patience = 0
        torch.save(model, model_path)
    else:
        patience += 1

    return patience, best_loss
