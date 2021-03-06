import torch
from torch.utils import data as dat

from torchvision import datasets
from torchvision import transforms

from matplotlib import pyplot as plt


def get_fashion_mnist_labels(labels):
    text_labels = ['T-shirt', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal',
                   'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    return [text_labels[int(s)] for s in labels]


def show_fashion_mnist(images, labels):
    _, ax = plt.subplots(nrows=1, ncols=len(images), figsize=(15, 5))
    for a, img, lbl in zip(ax, images, labels):
        a.imshow(img.view(28, 28).numpy())  # 绘制热图
        a.set_title(lbl)
        a.axes.get_xaxis().set_visible(False)  # 设置子图是否显示X轴
        a.axes.get_yaxis().set_visible(False)  # 设置子图是否显示Y轴
    plt.show()


def net(x_n):
    return softmax(torch.mm(x_n.view(-1, inputs), w) + b)


def softmax(x_s):
    x_exp = x_s.exp()
    partition = x_exp.sum(1, keepdim=True)
    return x_exp / partition


def cross_entropy(y_hat, y_true):
    return -torch.log(y_hat.gather(1, y_true.view(-1, 1)))


def sgd(params, l_r, b_s):
    for p in params:
        p.data -= l_r * p.grad / b_s


def accuracy(y_hat, y_true):
    return (y_hat.argmax(1) == y_true).float().mean().item()


def evaluate_accuracy(data_iter, network):
    acc_sum, n = 0.0, 0
    for x_i, y_i in data_iter:
        acc_sum += (network(x_i).argmax(1) == y_i).float().sum().item()
        n += y_i.shape[0]
    return acc_sum / n


# 获取并读取数据集
mnist_train = datasets.FashionMNIST(r'.\Datasets', train=True, transform=transforms.ToTensor(), download=True)
mnist_test = datasets.FashionMNIST(r'.\Datasets', train=False, transform=transforms.ToTensor(), download=True)
# print(type(mnist_train))
# print(len(mnist_train), len(mnist_test))

# feature, label = mnist_train[0]
# print(feature.shape)
# print(label)

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
inputs = 784
outputs = 10
w = torch.normal(0, 0.01, (inputs, outputs), requires_grad=True)
b = torch.zeros(outputs, requires_grad=True)

# 测试对多维Tensor按维度操作
# d = torch.tensor([[1, 2, 3], [4, 5, 6]])
# print(d.sum(0, keepdim=True))
# print(d.sum(1, keepdim=True))

# 测试softmax运算
# s = torch.randn(3, 5)
# x_prob = softmax(s)
# print(s)
# print(x_prob)
# print(x_prob.sum(1))

# 测试 gather 函数
# y_h = torch.tensor([[0.1, 0.3, 0.6], [0.3, 0.2, 0.5]])
# y_r = torch.tensor([0, 2])
# print(y_h.gather(1, y_r.view(-1, 1)))

# 测试 accuracy 函数
# print(accuracy(y_h, y_r))

# 测试 evaluate_accuracy 函数
# print(evaluate_accuracy(test_iter, net))

# 训练模型
epochs = 5
lr = 0.1
for epoch in range(epochs):
    train_loss, train_acc, count = 0.0, 0.0, 0
    for x, y in train_iter:
        y_pred = net(x)
        loss = cross_entropy(y_pred, y).sum()
        loss.backward()  # 反向传播
        sgd([w, b], lr, batch_size)  # 更新参数
        for param in [w, b]:
            param.grad.data.zero_()  # 清零梯度
        train_loss += loss.item()
        train_acc += (y_pred.argmax(1) == y).float().sum().item()
        count += y.shape[0]
    test_acc = evaluate_accuracy(test_iter, net)  # 测试集准确率
    print(f'epoch: {epoch + 1}, loss: {train_loss / count:.5f}, train acc: {train_acc / count:.5f}, test acc: {test_acc:.5f}')

# 预测测试集上一个批量样本
x_test, y_test = next(iter(test_iter))
labels_true = get_fashion_mnist_labels(y_test.numpy())
labels_pred = get_fashion_mnist_labels(net(x_test).argmax(1).detach().numpy())
title = [true + '\n' + pred for true, pred in zip(labels_true, labels_pred)]
show_fashion_mnist(x_test[0:9], title[0:9])
