import torch

from torch import nn
from torch import optim
from torch.nn import init
from torch.utils import data as dat

from matplotlib import pyplot as plt


# 定义模型
def linear(x_m, w_m, b_m):
    return torch.matmul(x_m, w_m) + b_m


# 定义损失函数
def squared_loss(y_h, y_t):
    return (y_h - y_t.view(y_h.size())) ** 2 / 2


# 定义优化算法
def sgd(params, l_r, b_s):
    for param in params:
        param.data -= l_r * param.grad / b_s


# 初始化模型参数
def init_params():
    w = torch.randn(inputs, 1, requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    return [w, b]


# 定义L2范数惩罚项
def l2_penalty(w):
    return (w ** 2).sum() / 2


def fit_and_plot(lbd):
    w, b = init_params()
    train_loss, test_loss = [], []
    for _ in range(epochs):
        for x, y in train_iter:
            los = (squared_loss(linear(x, w, b), y) + lbd * l2_penalty(w)).sum()
            los.backward()
            sgd([w, b], lr, batch_size)
            w.grad.data.zero_()
            b.grad.data.zero_()
        train_loss.append(squared_loss(linear(train_features, w, b), train_labels).mean().item())
        test_loss.append(squared_loss(linear(test_features, w, b), test_labels).mean().item())
    print(f'L2 norm of w: {w.data.norm().item()}')
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


# 生成数据
n_train, n_test, inputs = 20, 100, 200
true_w, true_b = torch.ones(inputs, 1) * 0.01, torch.tensor([0.05])
# print(true_w)
# print(true_b)

features = torch.randn(n_train + n_test, inputs)
labels = torch.matmul(features, true_w) + true_b
labels += torch.normal(0, 0.01, labels.size())

train_features, test_features = features[:n_train, :], features[n_train:, :]
train_labels, test_labels = labels[:n_train], labels[n_train:]

batch_size, epochs, lr = 1, 100, 0.003

dataset = dat.TensorDataset(train_features, train_labels)
train_iter = dat.DataLoader(dataset, batch_size, True)

# 过拟合
# fit_and_plot(0)

# 使用权重衰减
# fit_and_plot(3)


# ============================== 简洁实现 ============================== #
def fit_and_plot_pytorch(wd):
    net = nn.Linear(inputs, 1)
    init.normal_(net.weight, 0, 1)
    init.normal_(net.bias, 0, 1)
    loss = nn.MSELoss()
    optimizer_w = optim.SGD([net.weight], lr, weight_decay=wd)  # 对权重参数衰减
    optimizer_b = optim.SGD([net.bias], lr)  # 不对偏差参数衰减

    train_loss, test_loss = [], []
    for _ in range(epochs):
        for x, y in train_iter:
            los = loss(net(x), y).mean()
            los.backward()
            optimizer_w.step()
            optimizer_b.step()
            optimizer_w.zero_grad()
            optimizer_b.zero_grad()
        train_loss.append(loss(net(train_features), train_labels).mean().item())
        test_loss.append(loss(net(test_features), test_labels).mean().item())
    print(f'L2 norm of w: {net.weight.data.norm().item()}')
    plot(range(1, epochs + 1), train_loss, 'epochs', 'loss',
         range(1, epochs + 1), test_loss, ['train', 'test'])


# 过拟合
# fit_and_plot_pytorch(0)

# 使用权重衰减
# fit_and_plot_pytorch(3)
