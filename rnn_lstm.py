import torch

import numpy as np
import pandas as pd

from torch import nn, optim
from torch.nn import functional as F


# class Net(nn.Module):
#     def __init__(self, input_size):
#         super().__init__()
#         self.rnn = nn.LSTM(
#             input_size=input_size,
#             hidden_size=64,
#             num_layers=1,
#             bias=True,
#             batch_first=True,
#         )
#         self.out = nn.Sequential(
#             nn.Linear(64, 1)
#         )
#
#     def forward(self, x):
#         r_out, (h_n, h_c) = self.rnn(x, None)  # None 表示 hidden state 会用全0的 state
#         out = self.out(r_out[:, -1])
#         print(out.shape)


# cpst = pd.read_excel('成分17.9.17-12.31.xlsx', dtype=str)
# heat = pd.read_excel('加热实绩1710-12.xlsx', dtype=str)
# roll = pd.read_excel('轧钢实绩1710-12.xlsx', dtype=str)
# prop = pd.read_excel('机械性能_原始记录171001-180113.xlsx', dtype=str)

# print(cpst.shape)  # (9627, 57)
# print(heat.shape)  # (18768, 36)
# print(roll.shape)  # (18640, 49)
# print(prop.shape)  # (22107, 32)

# data = cpst.merge(roll, how='inner', on='plateid8').merge(heat, how='inner', on='plateid10').merge(prop, how='inner', on='plateid10')
# data = data.loc[(data['班别_x'] == data['班别_y']) &
#                 (data['班次_x'] == data['班次_y']) &
#                 (data['炉座号_x'] == data['炉座号_y']) &
#                 (data['板坯钢种_x'] == data['板坯钢种_y']) &
#                 (data['轧制钢种_x'] == data['轧制钢种_y']) &
#                 # (data['替代轧制_x'] == data['替代轧制_y']) &
#                 (data['板坯重量_x'] == data['板坯重量_y']) &
#                 (data['出炉时间_x'] == data['出炉时间_y']) &
#                 (data['是否紧急订单_x'] == data['是否紧急订单_y']) &
#                 (data['轧制异常'].isnull())]
# data.dropna(axis=1, how='all', inplace=True)
# data.reset_index(drop=True, inplace=True)
# data.to_excel('data.xlsx', index=False)

data = pd.read_excel('data.xlsx')
# print(data.loc[:, (data == data.loc[0]).all()])  # Se  Te  Zn  O  Mg  RE  Ta  ComA  ComC  ComH  ComI  ComL  ComV  ComP
cpst = data[['指示钢种', '钢种说明', '炼钢判定质量等级',
             'C', 'Mn', 'P', 'S', 'Si', 'Ni', 'Cr', 'Cu', 'Alt', 'Als', 'Nb', 'Mo',
             'V', 'Ti', 'Ca', 'Ceq', 'Pcm', 'Pb', 'Sb', 'Sn', 'As', 'B', 'W', 'Co',
             'Zr', 'N', 'H', 'Bi', 'ComB', 'ComD', 'ComE', 'ComF', 'ComG', 'ComJ', 'Com1',

             '']]


# lr = 0.01















