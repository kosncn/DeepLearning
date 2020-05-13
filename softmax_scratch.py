import time

import torch
import torchvision

import numpy as np

from torchvision import datasets
from torchvision import transforms

from torch.utils import data as dat
from matplotlib import pyplot as plt


def get_fashion_mnist_labels(labels):
    text_labels = ['T-shirt', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal',
                   'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    return [text_labels[int(s)] for s in labels]


def show_fashion_mnist(images, labels):
    _, ax = plt.subplots(1, len(images), figsize=(15, 5))
    for a, img, lbl in zip(ax, images, labels):
        a.imshow(img.view((28, 28)).numpy())  # 绘制热图
        a.set_title(lbl)
        a.axes.get_xaxis().set_visible(False)  # 设置子图是否显示X轴
        a.axes.get_yaxis().set_visible(False)  # 设置子图是否显示Y轴
    plt.show()


mnist_train = datasets.FashionMNIST(r'.\Datasets', train=True, transform=transforms.ToTensor(), download=True)
mnist_test = datasets.FashionMNIST(r'.\Datasets', train=False, transform=transforms.ToTensor(), download=True)
# print(type(mnist_train))
# print(len(mnist_train), len(mnist_test))

# feature, label = mnist_train[0]
# print(feature.shape, label)

# 查看训练集中前10个样本的图像内容和对应文本标签
# x_t, y_t = [], []
# for t in range(10):
#     x_t.append(mnist_train[t][0])
#     y_t.append(mnist_train[t][1])
# show_fashion_mnist(x_t, get_fashion_mnist_labels(y_t))

# 生成数据
batch_size = 256
train_iter = dat.DataLoader(mnist_train, batch_size=batch_size, shuffle=True)
test_iter = dat.DataLoader(mnist_test, batch_size=batch_size, shuffle=False)

# 初始化模型参数
num_inputs = 784
num_outputs = 10

W = torch.tensor(np.random.normal(0, 0.01, (num_inputs, num_outputs)), dtype=torch.float)
b = torch.zeros(num_outputs, dtype=torch.float)

W.requires_grad_(requires_grad=True)
b.requires_grad_(requires_grad=True)

# 实现softmax运算
X = torch.tensor([[1, 2, 3], [4, 5, 6]])
print(X.sum(dim=0, keepdim=True))
print(X.sum(dim=1, keepdim=True))


def softmax(X):
    X_exp = X.exp()
    partition = X_exp.sum(dim=1, keepdim=True)
    return X_exp / partition  # 这里应用了广播机制

X = torch.rand((2, 5))
X_prob = softmax(X)
print(X_prob, X_prob.sum(dim=1))


# 定义模型
def net(X):
    return softmax(torch.mm(X.view((-1, num_inputs)), W) + b)


# 定义损失函数
def cross_entropy(y_hat, y):
    return - torch.log(y_hat.gather(1, y.view(-1, 1)))


# 计算分类准确率
def accuracy(y_hat, y):
    return (y_hat.argmax(dim=1) == y).float().mean().item()


def evaluate_accuracy(data_iter, net):
    acc_sum, n = 0.0, 0
    for X, y in data_iter:
        acc_sum += (net(X).argmax(dim=1) == y).float().sum().item()
        n += y.shape[0]
    return acc_sum / n


# 训练模型
num_epochs, lr = 5, 0.1

# 本函数已保存在d2lzh包中方便以后使用
def train_ch3(net, train_iter, test_iter, loss, num_epochs, batch_size,
              params=None, lr=None, optimizer=None):
    for epoch in range(num_epochs):
        train_l_sum, train_acc_sum, n = 0.0, 0.0, 0
        for X, y in train_iter:
            y_hat = net(X)
            l = loss(y_hat, y).sum()

            # 梯度清零
            if optimizer is not None:
                optimizer.zero_grad()
            elif params is not None and params[0].grad is not None:
                for param in params:
                    param.grad.data.zero_()

            l.backward()
            if optimizer is None:
                d2l.sgd(params, lr, batch_size)
            else:
                optimizer.step()  # “softmax回归的简洁实现”一节将用到


            train_l_sum += l.item()
            train_acc_sum += (y_hat.argmax(dim=1) == y).sum().item()
            n += y.shape[0]
        test_acc = evaluate_accuracy(test_iter, net)
        print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f'
              % (epoch + 1, train_l_sum / n, train_acc_sum / n, test_acc))

train_ch3(net, train_iter, test_iter, cross_entropy, num_epochs, batch_size, [W, b], lr)

# 预测
X, y = iter(test_iter).next()

true_labels = d2l.get_fashion_mnist_labels(y.numpy())
pred_labels = d2l.get_fashion_mnist_labels(net(X).argmax(dim=1).numpy())
titles = [true + '\n' + pred for true, pred in zip(true_labels, pred_labels)]

d2l.show_fashion_mnist(X[0:9], titles[0:9])
