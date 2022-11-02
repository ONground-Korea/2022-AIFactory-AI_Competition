import warnings
import numpy as np
import wandb

from tqdm.auto import tqdm

from torch.utils.data import DataLoader

from glob import glob

import time

import torch
from torch import nn
import random
import torch.backends.cudnn as cudnn

from cosine_annealing_warmup import CosineAnnealingWarmupRestarts

from AIFactory.Transformer.data import windowDataset
from AIFactory.Transformer.helper import train_one_epoch, early_stopping
from AIFactory.Transformer.model import TFModel, Nong_metric

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
epochs = 100
device = torch.device("cuda")
data_root = 'C:/Users/hjs/OneDrive - 고려대학교/고려대학교/3-2/농산물/preprocessed'
outlier_root = 'C:/Users/hjs/OneDrive - 고려대학교/고려대학교/3-2/농산물/preprocessed/train_outlier'
data_list = glob(data_root + '/train/*.csv')
# data_list = glob(outlier_root + '/train_*.csv')
loss_arr = []
patiences = 20

########################################################################################################################


for data in (sorted(data_list)):
    # if idx != 34:
    #     continue

    df_number = data.split("_")[-1].split(".")[0]
    # print(df_number)
    # wandb.init(project="Nong", entity="onground", name=df_number, config={
    #     "epochs": epochs,
    # })

    lr = 1e-1
    model = TFModel(14, 28, 512, 8, 4, 0.1).to(device)
    criterion = nn.L1Loss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    # scheduler = CosineAnnealingWarmupRestarts(optimizer,
    #                                           first_cycle_steps=50,
    #                                           cycle_mult=1.5,
    #                                           max_lr=0.1,
    #                                           min_lr=1e-4,
    #                                           warmup_steps=10,
    #                                           gamma=0.5)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 0.95 ** epoch)

    train_dataset = windowDataset(data, is_val=True, phase='train')
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=False)
    val_dataset = windowDataset(data, is_val=True, phase='val')
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    total_loss, val_loss, best_loss = 0.0, 0.0, 1e10
    patience = 0
    st_time = time.time()
    progress = tqdm(range(epochs))
    for epoch in progress:
        total_loss, val_loss = train_one_epoch(epoch, train_loader, val_loader, device, criterion, optimizer, model,
                                               is_val=True)
        # wandb.log({"train_loss": total_loss, "val_loss": val_loss, "lr": optimizer.param_groups[0]['lr']})
        progress.set_description(
            f'[train_loss : {total_loss} / val_loss : {val_loss} / best_loss : {best_loss} / patience : {patience}] / lr : {optimizer.param_groups[0]["lr"]:.5f}')

        patience, best_loss = early_stopping(val_loss, patience, 0, best_loss, model, f'models/model_{df_number}.pth')
        if patience >= patiences:
            print(f'Early Stopping at {epoch} epoch')
            break
        scheduler.step()
    print(f'[{df_number}] final loss : {val_loss} / best loss : {best_loss} / time : {time.time() - st_time}')

    loss_arr.append((df_number, total_loss, val_loss, best_loss))
    # wandb.finish()
    # torch.save(model, f'models/model_{df_number}.pth')

import csv

# field names
fields = ['df_number', 'train_loss', 'val_loss', 'best_loss']

# data rows of csv file
rows = loss_arr

with open(f'loss_{epochs}_{patiences}.csv', 'w', newline='') as f:
    # using csv.writer method from CSV package
    write = csv.writer(f)

    write.writerow(fields)
    write.writerows(rows)

print(loss_arr)
