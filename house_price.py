import pandas as pd

from matplotlib import pyplot as plt

import torch
from torch import nn
from torch import optim
from torch.nn import init
from torch.utils import data as dat


# 定义生成与初始化模型函数
def get_net(num_features):
    net = nn.Linear(num_features, 1)
    for param in net.parameters():
        init.normal_(param, 0, 0.01)
    return net


# 定义对数均方误差函数
def rmse_loss(net, features, labels):
    with torch.no_grad():
        preds = torch.max(net(features), torch.tensor(1.0))
        rmse = torch.sqrt(loss(preds.log(), labels.log()))
    return rmse.item()


# 定义训练模型函数
def train(net, train_features, train_labels, test_features, test_labels, num_epochs, learning_rate, weight_decay, batch_size):
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
        train_loss.append(rmse_loss(net, train_features, train_labels))
        if test_labels is not None:
            test_loss.append(rmse_loss(net, test_features, test_labels))
    return train_loss, test_loss


# 定义生成第i折交叉验证所需数据函数
def get_k_fold_data(k, i, x, y):
    assert k > 1
    fold_size = x.shape[0] // k
    x_train, y_train = None, None
    x_valid, y_valid = None, None
    for j in range(k):
        index = slice(j * fold_size, (j + 1) * fold_size)
        x_part, y_part = x[index, :], y[index]
        if j == i:
            x_valid, y_valid = x_part, y_part
        elif x_train is None:
            x_train, y_train = x_part, y_part
        else:
            x_train = torch.cat((x_train, x_part), 0)
            y_train = torch.cat((y_train, y_part), 0)
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
        print(f'fold: {i}, train rmse: {train_loss[-1]:.5f}, valid rmse: {valid_loss[-1]:.5f}')
        if i == 0:
            plot(range(1, num_epochs + 1), train_loss, 'epochs', 'rmse',
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

    print(f'train rmse: {train_loss[-1]}')
    plot(range(1, num_epochs + 1), train_loss, 'epochs', 'rmse')

    preds = net(test_features).detach().numpy()
    test_data['SalePrice'] = pd.Series(preds.reshape(1, -1)[0])
    submission = pd.concat((test_data['Id'], test_data['SalePrice']), 1)
    submission.to_csv('submission.csv', index=False)
    print('result saved')


# 读取数据集
train_data = pd.read_csv('./Datasets/HousePrices/train.csv')
test_data = pd.read_csv('./Datasets/HousePrices/test.csv')
# print(train_data.shape)  # (1460, 81)
# print(test_data.shape)  # (1459, 80)
# print(train_data.iloc[0:5, [0, 1, 2, -3, -2, -1]])

all_features = pd.concat((train_data.iloc[:, 1:-1], test_data.iloc[:, 1:]), ignore_index=True)
# print(all_features.shape)  # (2919, 79)

# 预处理数据
numeric_features = all_features.dtypes.loc[all_features.dtypes != 'object'].index  # 获取数值列索引
all_features[numeric_features] = all_features[numeric_features].apply(lambda x: (x - x.mean()) / x.std())  # 标准化
all_features[numeric_features] = all_features[numeric_features].fillna(0)  # 处理缺失值
all_features = pd.get_dummies(all_features, dummy_na=True)  # one-hot
# print(all_features.shape)  # (2919, 331)

n_train = train_data.shape[0]
train_features = torch.tensor(all_features[:n_train].values, dtype=torch.float)
test_features = torch.tensor(all_features[n_train:].values, dtype=torch.float)
train_labels = torch.tensor(train_data['SalePrice'].values, dtype=torch.float).view(-1, 1)

# 训练模型
loss = nn.MSELoss()

k, num_epochs, learning_rate, weight_decay, batch_size = 5, 100, 5, 0, 64
train_loss_mean, valid_loss_mean = k_fold(k, train_features, train_labels, num_epochs, learning_rate, weight_decay, batch_size)
print(f'{k}-fold validation: avg train rmse {train_loss_mean:.5f}, avg valid rmse {valid_loss_mean:.5f}')

# 预测数据
# train_and_pred(train_features, train_labels, test_features, test_data, num_epochs, learning_rate, weight_decay, batch_size)
