import torch
from torch import nn
from torch.utils import data as dat

from torchvision import datasets
from torchvision import transforms

from matplotlib import pyplot as plt


# 定义模型
def net(x_n):
    h = relu(torch.matmul(x_n.view(-1, inputs), w1) + b1)
    return torch.matmul(h, w2) + b2


# 定义激活函数
def relu(x_r):
    return torch.max(x_r, torch.tensor(0.0))


# 定义更新函数
def sgd(params, l_r):
    for p in params:
        p.data -= l_r * p.grad


def evaluate_accuracy(data_iter, network):
    acc_sum, n = 0.0, 0
    for x_i, y_i in data_iter:
        acc_sum += (network(x_i).argmax(1) == y_i).float().sum().item()
        n += y_i.shape[0]
    return acc_sum / n


def plot(x_vals, y_vals, name):
    plt.figure(figsize=(7, 3.5))
    plt.xlabel('x')
    plt.ylabel(name + 'y')
    plt.plot(x_vals.detach().numpy(), y_vals.detach().numpy())
    plt.show()


# ReLU函数
# x_c = torch.arange(-10.0, 10.0, 0.1, requires_grad=True)
# y_r = x_c.relu()
# plot(x_c, y_r, 'relu')

# ReLU函数的导数
# y_r.sum().backward()
# plot(x_c, x_c.grad, 'grad of relu')
# x_c.grad.zero_()

# Sigmoid函数
# y_s = x_c.sigmoid()
# plot(x_c, y_s, 'sigmoid')

# Sigmoid函数的导数
# y_s.sum().backward()
# plot(x_c, x_c.grad, 'grad of sigmoid')
# x_c.grad.zero_()

# tanh函数
# y_t = x_c.tanh()
# plot(x_c, y_t, 'tanh')

# tanh函数的导数
# y_t.sum().backward()
# plot(x_c, x_c.grad, 'grad of tanh')
# x_c.grad.zero_()

# 获取并读取数据集
mnist_train = datasets.FashionMNIST(r'.\Datasets', train=True, transform=transforms.ToTensor(), download=True)
mnist_test = datasets.FashionMNIST(r'.\Datasets', train=False, transform=transforms.ToTensor(), download=True)

# 生成数据
batch_size = 256
train_iter = dat.DataLoader(mnist_train, batch_size, True)
test_iter = dat.DataLoader(mnist_test, batch_size, False)

# 定义模型参数
inputs = 784
outputs = 10
hiddens = 256
w1 = torch.normal(0, 0.01, (inputs, hiddens), requires_grad=True)
b1 = torch.zeros(hiddens, requires_grad=True)
w2 = torch.normal(0, 0.01, (hiddens, outputs), requires_grad=True)
b2 = torch.zeros(outputs, requires_grad=True)

# 定义损失函数
loss = nn.CrossEntropyLoss()

# 训练模型
epochs = 5
lr = 0.5
for epoch in range(epochs):
    train_loss, train_acc, count = 0.0, 0.0, 0
    for x, y in train_iter:
        y_pred = net(x)
        los = loss(y_pred, y).sum()
        los.backward()
        sgd([w1, b1, w2, b2], lr)
        for param in [w1, b1, w2, b2]:
            param.grad.data.zero_()
        train_loss += los.item()
        train_acc += (y_pred.argmax(1) == y).float().sum().item()
        count += y.shape[0]
    test_acc = evaluate_accuracy(test_iter, net)  # 测试集准确率
    print(f'epoch: {epoch + 1}, loss: {train_loss / count:.5f}, train acc: {train_acc / count:.5f}, test acc: {test_acc:.5f}')
