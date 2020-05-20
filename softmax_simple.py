from torch import nn
from torch import optim
from torch.nn import init
from torch.utils import data as dat

from torchvision import datasets
from torchvision import transforms

from collections import OrderedDict

from matplotlib import pyplot as plt


class FlattenLayer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x_m):  # x shape: (batch_size, *, *, *)
        return x_m.view(x_m.shape[0], -1)


class LinearNet(nn.Module):
    def __init__(self, ipt, opt):
        super().__init__()
        self.linear = nn.Linear(ipt, opt)

    def forward(self, x_m):
        y_m = self.linear(x_m.view(x_m.shape[0], -1))  # x shape: (batch_size, 1, 28, 28)
        return y_m


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


def evaluate_accuracy(data_iter, network):
    acc_sum, n = 0.0, 0
    for x_i, y_i in data_iter:
        acc_sum += (network(x_i).argmax(1) == y_i).float().sum().item()
        n += y_i.shape[0]
    return acc_sum / n


# 下载并读取数据集
mnist_train = datasets.FashionMNIST(r'.\Datasets', train=True, transform=transforms.ToTensor(), download=True)
mnist_test = datasets.FashionMNIST(r'.\Datasets', train=False, transform=transforms.ToTensor(), download=True)

# 生成数据
batch_size = 256
train_iter = dat.DataLoader(mnist_train, batch_size=batch_size, shuffle=True)
test_iter = dat.DataLoader(mnist_test, batch_size=batch_size, shuffle=False)

# 初始化模型参数
inputs = 784
outputs = 10

# 定义模型（方式一）
# net = LinearNet(inputs, outputs)

# 定义模型（方式二）
net = nn.Sequential(
    # FlattenLayer(),
    # nn.Linear(inputs, outputs)

    OrderedDict([
        ('flatten', FlattenLayer()),
        ('linear', nn.Linear(inputs, outputs))
    ])
)

# 初始化模型权重参数
init.normal_(net.linear.weight, 0, 0.01)
init.constant_(net.linear.bias, 0)

# 定义损失函数
loss = nn.CrossEntropyLoss()

# 定义优化算法
optimizer = optim.SGD(net.parameters(), 0.1)

# 训练模型
epochs = 5

for epoch in range(epochs):
    train_loss, train_acc, count = 0.0, 0.0, 0
    for x, y in train_iter:
        y_hat = net(x)
        los = loss(y_hat, y).sum()
        los.backward()  # 反向传播
        optimizer.step()  # 更新参数
        optimizer.zero_grad()  # 清零梯度
        train_loss += los.item()
        train_acc += (y_hat.argmax(1) == y).float().sum().item()
        count += y.shape[0]
    test_acc = evaluate_accuracy(test_iter, net)
    print(f'epoch: {epoch + 1}, loss: {train_loss / count:.5f}, train acc: {train_acc / count:.5f}, test acc: {test_acc:.5f}')

# 预测测试集上一个批量样本
x_test, y_test = next(iter(test_iter))
labels_true = get_fashion_mnist_labels(y_test.numpy())
labels_pred = get_fashion_mnist_labels(net(x_test).argmax(1).detach().numpy())
title = [true + '\n' + pred for true, pred in zip(labels_true, labels_pred)]
show_fashion_mnist(x_test[0:9], title[0:9])
