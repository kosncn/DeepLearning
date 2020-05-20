import torch

from torch import nn
from torch import optim
from torch.utils import data as dat

from matplotlib import pyplot as plt


def fit_and_plot(train_x, test_x, train_y, test_y):
    net = nn.Linear(train_x.shape[-1], 1)

    batch_size = min(10, train_y.shape[0])
    dataset = dat.TensorDataset(train_x, train_y)
    train_iter = dat.DataLoader(dataset, batch_size, True)

    optimizer = optim.SGD(net.parameters(), 0.01)

    train_loss, test_loss = [], []
    for _ in range(epochs):
        for x, y in train_iter:
            los = loss(net(x), y.view(-1, 1))
            los.backward()
            optimizer.step()
            optimizer.zero_grad()
        train_y = train_y.view(-1, 1)
        test_y = test_y.view(-1, 1)
        train_loss.append(loss(net(train_x), train_y).item())
        test_loss.append(loss(net(test_x), test_y).item())
    print(f'final epoch - train loss: {train_loss[-1]}, test loss: {test_loss[-1]}')
    print(f'weight: {net.weight.data}\nbias: {net.bias.data}')
    plot(range(1, epochs + 1), train_loss, 'epochs', 'loss',
         range(1, epochs + 1), test_loss, ['train', 'test'])


def plot(x_vals, y_vals, x_label, y_label, x2_vals=None, y2_vals=None, legend=None, figsize=(7, 5)):
    plt.figure(figsize=figsize)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.semilogy(x_vals, y_vals)
    if x2_vals and y2_vals:
        plt.semilogy(x2_vals, y2_vals, linestyle=':')
        plt.legend(legend)
    plt.show()


# 生成数据集
n_train, n_test, true_w, true_b = 100, 100, [1.2, -3.1, 5.6], 5

x_part = torch.randn(n_train + n_test, 1)
x_total = torch.cat((x_part, torch.pow(x_part, 2), torch.pow(x_part, 3)), 1)

labels = true_w[0] * x_total[:, 0] + true_w[1] * x_total[:, 1] + true_w[2] * x_total[:, 2] + true_b
labels += torch.normal(0, 0.01, labels.size())

# print(x_part[:3])
# print(x_total[:3])
# print(labels[:3])

epochs = 100
loss = nn.MSELoss()

# 三阶多项式函数拟合
# fit_and_plot(x_total[:n_train], x_total[n_train:], labels[:n_train], labels[n_train:])

# 线性函数拟合（欠拟合）
# fit_and_plot(x_part[:n_train], x_part[n_train:], labels[:n_train], labels[n_train:])

# 训练样本不足（过拟合）
fit_and_plot(x_total[:2, :], x_total[n_train:, :], labels[:2], labels[n_train:])
