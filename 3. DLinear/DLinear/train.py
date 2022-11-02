import warnings
import numpy as np

from tqdm.auto import tqdm

import torch.nn as nn
from torch.utils.data import DataLoader

from glob import glob

import time

import torch
import random
import torch.backends.cudnn as cudnn

from AIFactory.DLinear.data import windowDataset
from AIFactory.DLinear.helper import train_one_epoch, early_stopping
from AIFactory.DLinear.model import DLinear

torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.cuda.manual_seed_all(42)
np.random.seed(42)
cudnn.benchmark = False
cudnn.deterministic = True
random.seed(42)
warnings.filterwarnings(action='ignore')

# parameters
########################################################################################################################
epochs = 250
device = torch.device("cuda")
data_root = 'C:/Users/hjs/PycharmProjects/pytorch/AIFactory/preprocess2'
data_list = glob(data_root + '/train/*.csv')
# data_list = glob(outlier_root + '/train_*.csv')
loss_arr = []
is_val = False
########################################################################################################################


for idx in range(0, 37):
    data_list = glob(f'{data_root}/train/train_{idx}.csv')[0]
    # if idx != 34:
    #     continue

    df_number = data_list.split("_")[-1].split(".")[0]
    print(df_number)
    # wandb.init(project="Nong", entity="onground", name=df_number, config={
    #     "epochs": epochs,
    # })

    lr = 1e-2
    model = DLinear(14, 28).to(device)
    criterion = nn.L1Loss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 0.99 ** epoch, verbose=False)

    train_dataset = windowDataset(data_list, is_val=is_val, phase='train')
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=False)
    if is_val:
        val_dataset = windowDataset(data_list, is_val=is_val, phase='val')
        val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
    else:
        val_dataset = None
        val_loader = None

    total_loss, val_loss, best_loss = 0.0, 0.0, 1e10
    patience = 0
    st_time = time.time()
    progress = tqdm(range(epochs))
    for epoch in progress:
        total_loss, val_loss = train_one_epoch(epoch, train_loader, val_loader, device, criterion, optimizer, model,
                                               is_val=is_val)
        # wandb.log({"train_loss": total_loss, "val_loss": val_loss, "lr": optimizer.param_groups[0]['lr']})
        progress.set_description(
            f'[train_loss : {total_loss} / val_loss : {val_loss} / best_loss : {best_loss} / patience : {patience}] / lr : {optimizer.param_groups[0]["lr"]:.5f}')

        patience, best_loss = early_stopping(val_loss, patience, 0, best_loss, model, f'models/model_{df_number}.pth')
        if patience >= 50:
            print(f'Early Stopping at {epoch} epoch')
            break
        scheduler.step()
    print(f'[{df_number}] final loss : {val_loss} / best loss : {best_loss} / time : {time.time() - st_time}')

    loss_arr.append((df_number, total_loss, val_loss, best_loss))
    # wandb.finish()
    # torch.save(model, f'models/model_{df_number}.pth')
    if not (is_val):
        torch.save(model, f'models/model_{df_number}.pth')

print(loss_arr)
