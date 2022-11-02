import os
import pandas as pd
import numpy as np

from tqdm.auto import tqdm

from torch.utils.data import DataLoader

from glob import glob

import torch
import random
import torch.backends.cudnn as cudnn

from AIFactory.Transformer.data import testDataset

torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.cuda.manual_seed_all(42)
np.random.seed(42)
cudnn.benchmark = False
cudnn.deterministic = True
random.seed(42)

for i in tqdm(range(10)):
    data_list = glob(f'C:/Users/hjs/OneDrive - 고려대학교/고려대학교/3-2/농산물/preprocessed/test/set_{i}/*.csv')

    for idx, data in enumerate(sorted(data_list)):
        file_number = data.split('test_')[1].split('.')[0]
        # print(file_number)
        test_dataset = testDataset(data)
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

        try:
            model = torch.load(f'models/model_{file_number}.pth')
        except:
            print('no model!!')

        model.eval()
        for (inputs, outputs) in test_loader:
            inputs = inputs.to('cuda')
            pred = model(inputs.float())
            pred = pred.to('cpu').detach().numpy()

        # model test
        if os.path.exists('./model_output') == False:
            os.mkdir('./model_output')

        if os.path.exists(f'./model_output/set_{i}') == False:
            os.mkdir(f'./model_output/set_{i}')

        # 결과 저장
        save_df = pd.DataFrame(pred).T
        save_df.to_csv(f'./model_output/set_{i}/predict_{file_number}.csv', index=False)

for k in tqdm(range(10)):

    globals()[f'set_df_{k}'] = pd.DataFrame()
    answer_df_list = glob(f'./model_output/set_{k}/*.csv')  # 예측한 결과 불러오기
    pum_list = glob(
        f'C:/Users/hjs/OneDrive - 고려대학교/고려대학교/3-2/농산물/aT_data/aT_test_raw/sep_{k}/*.csv')  # 기존 test input 불러오기
    pummok = [a for a in pum_list if 'pummok' in a.split('/')[-1]]

    for i in answer_df_list:
        df = pd.read_csv(i)
        number = i.split('_')[-1].split('.')[0]

        base_number = 0
        for p in pummok:
            if number == p.split('_')[-1].split('.')[0]:
                pum_df = pd.read_csv(p)

                if len(pum_df) != 0:
                    base_number = pum_df.iloc[len(pum_df) - 1][
                        '해당일자_전체평균가격(원)']  # 기존 각 sep 마다 test input의 마지막 target 값 가져오기 (변동률 계산을 위해)
                else:
                    base_number = np.nan

        globals()[f'set_df_{k}'][f'품목{number}'] = [base_number] + list(
            df[df.columns[-1]].values)  # 각 품목당 순서를 t, t+1 ... t+28 로 변경

    globals()[f'set_df_{k}'] = globals()[f'set_df_{k}'][[f'품목{col}' for col in range(37)]]  # 열 순서를 품목0 ~ 품목36 으로 변경

date = [f'd+{i}' for i in range(1, 15)] + ['d+22 ~ 28 평균']

for k in range(10):
    globals()[f'answer_df_{k}'] = pd.DataFrame()
    for c in globals()[f'set_df_{k}'].columns:
        base_d = globals()[f'set_df_{k}'][c][0]  # 변동률 기준 t 값

        ans_1_14 = []
        for i in range(14):
            ans_1_14.append(globals()[f'set_df_{k}'][c].iloc[i + 1])  # t+1 ~ t+14 까지는 (t+n - t)/t 로 계산

        ans_22_28 = (globals()[f'set_df_{k}'][c][22:29].mean())  # t+22 ~ t+28은 np.mean(t+22 ~ t+28) - t / t

        globals()[f'answer_df_{k}'][f'{c} 변동률'] = ans_1_14 + [ans_22_28]

    globals()[f'answer_df_{k}']['Set'] = k  # set 번호 설정
    globals()[f'answer_df_{k}']['일자'] = date  # 일자 설정

# 위에서 계산된 변동률 들을 합쳐주는 과정

all_df = pd.DataFrame()
for i in range(10):
    if i == 0:
        all_df = pd.concat([all_df, globals()[f'answer_df_{i}']], axis=1)
    else:
        all_df = pd.concat([all_df, globals()[f'answer_df_{i}']])

all_df = all_df[['Set', '일자'] + list(all_df.columns[:-2])]
all_df.reset_index(drop=True, inplace=True)

# set, 일자 기억하기위해 따로 저장

re_set = list(all_df['Set'])
re_date = list(all_df['일자'])

# 정답 양식 불러오기
out_ans = pd.read_csv('answer_example.csv')

# 두 dataframe 합치기 (nan + 숫자 = nan 이용)
submit_df = all_df + out_ans

submit_df['Set'] = re_set
submit_df['일자'] = re_date

# 최종 저장
submit_df.to_csv('./submit.csv', index=False, encoding='utf-8-sig')
