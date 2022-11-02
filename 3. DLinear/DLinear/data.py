import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from tqdm.auto import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn import Transformer

import math
from glob import glob
import torch
import random
import torch.backends.cudnn as cudnn

torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.cuda.manual_seed_all(42)
np.random.seed(42)
cudnn.benchmark = False
cudnn.deterministic = True
random.seed(42)

tr_del_list = ['단가(원)', '거래량', '거래대금(원)', '경매건수', '도매시장코드', '도매법인코드', '산지코드 ', '일자구분_중순', '일자구분_초순', '일자구분_하순',
               '월구분_10월', '월구분_11월', '월구분_12월', '월구분_1월', '월구분_2월', '월구분_3월',
               '월구분_4월', '월구분_5월', '월구분_6월', '월구분_7월', '월구분_8월', '월구분_9월']  # train 에서 사용하지 않는 열
ts_del_list = ['단가(원)', '거래량', '거래대금(원)', '경매건수', '도매시장코드', '도매법인코드', '산지코드 ', '일자구분_중순', '일자구분_초순', '일자구분_하순',
               '월구분_10월', '월구분_11월', '월구분_12월', '월구분_1월', '월구분_2월', '월구분_3월',
               '월구분_4월', '월구분_5월', '월구분_6월', '월구분_7월', '월구분_8월', '월구분_9월']  # test 에서 사용하지 않는 열
check_col = ['일자구분_중순', '일자구분_초순', '일자구분_하순', '월구분_10월', '월구분_11월', '월구분_12월', '월구분_1월', '월구분_2월', '월구분_3월',
             '월구분_4월', '월구분_5월', '월구분_6월', '월구분_7월', '월구분_8월', '월구분_9월']  # 열 개수 맞추기


def time_feature(dates, freq='h'):
    dates['month'] = dates.date.apply(lambda row: row.month, 1)
    dates['day'] = dates.date.apply(lambda row: row.day, 1)
    dates['weekday'] = dates.date.apply(lambda row: row.weekday(), 1)
    dates['hour'] = dates.date.apply(lambda row: row.hour, 1)
    dates['minute'] = dates.date.apply(lambda row: row.minute, 1)
    dates['minute'] = dates.minute.map(lambda x: x // 15)
    freq_map = {
        'y': [], 'm': ['month'], 'w': ['month'], 'd': ['month', 'day', 'weekday'],
        'b': ['month', 'day', 'weekday'], 'h': ['month', 'day', 'weekday', 'hour'],
        't': ['month', 'day', 'weekday', 'hour', 'minute'],
    }
    return dates[freq_map[freq.lower()]].values


def time_window(df, t, t_sep):
    """
    making time window, same as AIFactory baseline code
    :param df:
    :param t:
    :param t_sep:
    :return:
    """
    seq_len = t
    seqence_length = seq_len + t_sep

    result = []
    for index in range(len(df) - seqence_length):
        tmp = df[index: index + seqence_length]
        tmp = np.vstack(tmp).astype(np.float)
        tmp = torch.from_numpy(tmp)
        result.append(tmp)

    return np.array(result)


def make_dataset(i, is_val=False, phase=None):
    """
    make data for train, val
    :param i: data root (path)
    :param is_val: if True, split train, val
    :param phase: train, val phase. if phase is None, return whole data
    :return: xdata, ydata, x_ratio. x_ratio is used for calculate loss(for calculating change rate in helper.py/train_one_epoch)
    """
    df_number = i.split("_")[-1].split(".")[0]
    df = pd.read_csv(i)

    for j in df.columns:
        df[j] = df[j].replace({' ': np.nan})

    # 사용할 열 선택 및 index 설정
    df.drop(tr_del_list, axis=1, inplace=True)
    df.set_index('datadate', drop=False, inplace=True)

    # nan 처리
    df = df.fillna(0)

    # 변수와 타겟 분리
    x, y = df[[i for i in df.columns if i == '해당일자_전체평균가격(원)' or i == 'datadate']], df['해당일자_전체평균가격(원)']
    x_ratio = df['해당일자_전체평균가격(원)']

    # 2주 입력을 통한 이후 4주 예측을 위해 y의 첫 14일을 제외
    y = y[14:]

    x['datadate'] = pd.to_datetime(x['datadate'].astype('str'))
    x_stamp = df_stamp = pd.DataFrame(columns=['date'])
    x_stamp.date = list(x['datadate'].values)
    x_stamp = time_feature(x_stamp, freq='d')

    x.drop('datadate', axis=1, inplace=True)
    x = x.values

    # time series window 생성
    data_x = time_window(x, 13, 1)
    data_y = time_window(y, 27, 1)
    data_x_ratio = time_window(x_ratio, 41, 1)

    # y의 길이와 같은 길이로 설정
    xdata = data_x[:len(data_y)]
    ydata = data_y
    x_ratio = data_x_ratio

    # train, val 분리
    if is_val:
        x_train, x_val, y_train, y_val = train_test_split(xdata, ydata, test_size=0.3, shuffle=False, random_state=42)
        x_train, x_val, y_train_ratio, y_train_val = train_test_split(xdata, x_ratio, test_size=0.3, shuffle=False,
                                                                      random_state=42)

        if phase == 'train':
            xdata = x_train
            ydata = y_train
            x_ratio = y_train_ratio

        elif phase == 'val':
            xdata = x_val
            ydata = y_val
            x_ratio = y_train_val

        del x_train, x_val, y_train, y_val, y_train_ratio, y_train_val

    return xdata, ydata, x_ratio


class windowDataset(Dataset):
    def __init__(self, data, is_val=False, phase=None):
        """
        making dataset
        :param data:
        :param is_val:
        :param phase:
        """
        if is_val:
            self.xdata, self.ydata, self.x_ratio = make_dataset(data, is_val, phase)

        else:
            self.xdata, self.ydata, self.x_ratio = make_dataset(data)

    def __len__(self):
        return len(self.xdata)

    def __getitem__(self, idx):
        return self.xdata[idx], self.ydata[idx].reshape(-1), self.x_ratio[idx]


def make_Tensor(array):
    return torch.from_numpy(array)


def astype_data(data):
    df = data.astype(np.float32)
    return make_Tensor(df)


class testDataset(Dataset):
    def __init__(self, data):
        zero_csv = [0 for i in range(14)]
        df = pd.read_csv(data)

        if len(df) == 0:
            df['zero_non'] = zero_csv
            df = df.fillna(0)
            df.drop('zero_non', axis=1, inplace=True)
            df.drop('Unnamed: 0', axis=1, inplace=True)

        file_number = data.split('test_')[1].split('.')[0]

        # 사용할 열 선택, index 설정
        df.set_index('datadate', drop=False, inplace=True)

        # train input 과 형상 맞추기
        add_col = [i for i in check_col if i not in df.columns]

        for a in add_col:
            df[a] = 0

        df.drop(ts_del_list, axis=1, inplace=True)

        # ' ' -> nan 으로 변경
        for a in df.columns:
            df[a] = df[a].replace({' ': np.nan})

        # nan 처리
        df = df.fillna(0)
        df = df[[i for i in df.columns if i == '해당일자_전체평균가격(원)' or i == 'datadate']]

        df['datadate'] = pd.to_datetime(df['datadate'].astype('str'))
        df_stamp = pd.DataFrame(columns=['date'])
        df_stamp.date = list(df['datadate'].values)
        df_stamp = time_feature(df_stamp, freq='d')

        df.drop('datadate', axis=1, inplace=True)
        df = df.values

        # x_test  생성
        self.df_test = astype_data(df.reshape(1, df.shape[0], df.shape[1]))

    def __len__(self):
        return len(self.df_test)

    def __getitem__(self, idx):
        return self.df_test[idx], self.df_test[idx]
