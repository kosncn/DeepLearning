import pandas as pd

from matplotlib import pyplot as plt

import torch
from torch import nn
from torch import optim
from torch.nn import init
from torch.utils import data as dat


# 定义生成与初始化模型函数
def get_net(num_features):
    # net = nn.Linear(num_features, 3)  # (222, 3)

    net = nn.Sequential(
        nn.Linear(num_features, 500),
        nn.ReLU(),
        nn.Linear(500, 250),
        nn.ReLU(),
        nn.Linear(250, 3)
    )

    for param in net.parameters():
        init.normal_(param, 0, 0.01)
    return net


# 定义训练模型函数
def train(net, train_features, train_labels, test_features, test_labels, num_epochs, learning_rate, weight_decay, batch_size):
    # print(f'train 1: {train_features.shape}')  # torch.Size([12216, 222])
    # print(f'train 2: {train_labels.shape}')  # torch.Size([12216, 3])
    # print(f'train 3: {test_features.shape}')  # torch.Size([3054, 222])
    # print(f'train 4: {test_labels.shape}')  # torch.Size([3054, 3])

    # 生成数据
    dataset = dat.TensorDataset(train_features, train_labels)
    train_iter = dat.DataLoader(dataset, batch_size, True)

    # 定义优化算法
    optimizer = optim.Adam(net.parameters(), learning_rate, weight_decay=weight_decay)

    # 训练模型
    train_loss, test_loss = [], []
    for _ in range(num_epochs):
        for x, y in train_iter:
            los = loss(net(x), y)
            optimizer.zero_grad()
            los.backward()
            optimizer.step()
        train_loss.append(loss(net(train_features), train_labels))
        if test_labels is not None:
            test_loss.append(loss(net(test_features), test_labels))
    return train_loss, test_loss


# 定义生成第i折交叉验证所需数据函数
def get_k_fold_data(k, i, x, y):
    # print(f'get_k_fold_data 1: {x.shape}')  # torch.Size([15270, 222])
    # print(f'get_k_fold_data 2: {y.shape}')  # torch.Size([15270, 3])

    assert k > 1
    fold_size = x.shape[0] // k
    x_train, y_train = None, None
    x_valid, y_valid = None, None
    for j in range(k):
        index = slice(j * fold_size, (j + 1) * fold_size)
        x_part, y_part = x[index, :], y[index, :]
        if j == i:
            x_valid, y_valid = x_part, y_part
        elif x_train is None:
            x_train, y_train = x_part, y_part
        else:
            x_train = torch.cat((x_train, x_part), 0)
            y_train = torch.cat((y_train, y_part), 0)

        # if j == 0:
            # print(f'get_k_fold_data 3: {index}')  # slice(0, 3054, None)
            # print(f'get_k_fold_data 4: {x_part.shape}')  # torch.Size([3054, 222])
            # print(f'get_k_fold_data 5: {y_part.shape}')  # torch.Size([3054, 3])

    return x_train, y_train, x_valid, y_valid


# 定义k折交叉验证函数
def k_fold(k, x_train, y_train, num_epochs, learning_rate, weight_decay, batch_size):
    train_loss_sum, valid_loss_sum = 0, 0
    for i in range(k):
        data = get_k_fold_data(k, i, x_train, y_train)
        net = get_net(x_train.shape[1])
        train_loss, valid_loss = train(net, *data, num_epochs, learning_rate, weight_decay, batch_size)
        train_loss_sum += train_loss[-1]
        valid_loss_sum += valid_loss[-1]

        # if i == 0:
        #     print(f'k_fold 1: {type(data)}')  # <class 'tuple'>
        #     print(f'k_fold 2: {net.modules()}')  # <generator object Module.modules at 0x0000017B2CBFD148>

        print(f'fold: {i}, train mse: {train_loss[-1]:.5f}, valid mse: {valid_loss[-1]:.5f}')
        if i == 0:
            plot(range(1, num_epochs + 1), train_loss, 'epochs', 'mse',
                 range(1, num_epochs + 1), valid_loss, ['train', 'valid'])
    return train_loss_sum / k, valid_loss_sum / k


def plot(x_vals, y_vals, x_label, y_label, x2_vals=None, y2_vals=None, legend=None, figsize=(7, 5)):
    plt.figure(figsize=figsize)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.semilogy(x_vals, y_vals)
    if x2_vals and y2_vals:
        plt.semilogy(x2_vals, y2_vals, linestyle=':')
        plt.legend(legend)
    plt.show()


def train_and_pred(train_features, train_labels, test_features, test_data, num_epochs, learning_rate, weight_decay, batch_size):
    net = get_net(train_features.shape[1])
    train_loss, _ = train(net, train_features, train_labels, None, None, num_epochs, learning_rate, weight_decay, batch_size)

    print(f'train mse: {train_loss[-1]}')
    plot(range(1, num_epochs + 1), train_loss, 'epochs', 'mse')

    # preds = net(test_features).detach().numpy()
    # test_data['SalePrice'] = pd.Series(preds.reshape(1, -1)[0])
    # submission = pd.concat((test_data['Id'], test_data['SalePrice']), 1)
    # submission.to_csv('submission.csv', index=False)
    # print('result saved')


# 读取数据
cpst = pd.read_excel('成分17.9.17-12.31.xlsx')
heat = pd.read_excel('加热实绩1710-12.xlsx')
roll = pd.read_excel('轧钢实绩1710-12.xlsx')
prop = pd.read_excel('机械性能_原始记录171001-180113.xlsx')
# print(cpst.shape)  # (9627, 57)
# print(heat.shape)  # (18768, 36)
# print(roll.shape)  # (18640, 49)
# print(prop.shape)  # (22107, 32)

# 合并数据
cpst['plateid8'] = cpst['plateid8'].astype('str')  # 将成分表中'plateid8'列的类型由int64转换为object
data = cpst.merge(roll, how='inner', on='plateid8').merge(heat, how='inner', on='plateid10').merge(prop, how='inner', on='plateid10')
data = data.loc[(data['班别_x'] == data['班别_y']) &
                (data['班次_x'] == data['班次_y']) &
                (data['炉座号_x'] == data['炉座号_y']) &
                (data['板坯钢种_x'] == data['板坯钢种_y']) &
                (data['轧制钢种_x'] == data['轧制钢种_y']) &
                # (data['替代轧制_x'] == data['替代轧制_y']) &
                (data['板坯重量_x'] == data['板坯重量_y']) &
                (data['出炉时间_x'] == data['出炉时间_y']) &
                (data['是否紧急订单_x'] == data['是否紧急订单_y']) &
                (data['轧制异常'].isnull())]

# 处理异常数据（'板坯钢种_x', '标准钢种'）
data['板坯钢种_x'] = data['板坯钢种_x'].astype('str')
data = data.loc[data['板坯钢种_x'].apply(lambda x: len(x) < 50)]

data = data.loc[data['yield'].notnull()]

# 提取数据
data_features = data[['C', 'Mn', 'P', 'S', 'Si', 'Ni', 'Cr', 'Cu', 'Alt', 'Als', 'Nb', 'Mo',
                      'V', 'Ti', 'Ca', 'Ceq', 'Pcm', 'Pb', 'Sb', 'Sn', 'As', 'B', 'W', 'Co',
                      'Se', 'Te', 'Zn', 'Zr', 'N', 'O', 'H', 'Mg', 'Bi',
                      '炉座号_x', '布料方式', '在炉时间', '装炉温度', '出炉温度',
                      '轧制方式', '板坯钢种_x', '标准钢种', '厚度', '宽度', '长度',
                      '轧制厚度', '二阶段温度', '待温厚度比', '终轧温度', '返红温度', '纯轧时间（分）']].copy()
data_labels = data[['yield', 'T.S\nMPa', 'A\n%']].copy()

index = data_features.dtypes.loc[data_features.dtypes != 'object'].index  # 获取数值列索引
data_features[index] = data_features[index].apply(lambda x: (x - x.mean()) / x.std())  # 标准化
data_features[index] = data_features[index].fillna(0)  # 填补缺失值为0

# one-hot
data_features = pd.get_dummies(data_features, dummy_na=True)
# print(data_features.shape)  # (19009, 220)

n_train = int(data_features.shape[0] * 0.8)
train_samples = torch.tensor(data_features[:n_train].values, dtype=torch.float)
test_samples = torch.tensor(data_features[n_train:].values, dtype=torch.float)
train_labels = torch.tensor(data_labels[:n_train].values, dtype=torch.float)
test_labels = torch.tensor(data_labels[n_train:].values, dtype=torch.float)

# print(train_samples.shape)  # torch.Size([15207, 220])
# print(test_samples.shape)  # torch.Size([3802, 220])
# print(train_labels.shape)  # torch.Size([15207, 3])

# # 定义损失函数
loss = nn.MSELoss()

k, num_epochs, learning_rate, weight_decay, batch_size = 10, 100, 0.03, 0, 100
train_loss_mean, valid_loss_mean = k_fold(k, train_samples, train_labels, num_epochs, learning_rate, weight_decay, batch_size)
print(f'{k}-fold validation: avg train mse {train_loss_mean:.5f}, avg valid mse {valid_loss_mean:.5f}')

