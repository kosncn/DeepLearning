import time

import torch
import torchvision

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




