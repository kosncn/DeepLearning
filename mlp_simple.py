import torch

from torch import nn
from torch import optim
from torch.nn import init
from torch.utils import data as dat

from torchvision import datasets
from torchvision import transforms


class FlattenLayer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x_m):
        return x_m.view(x_m.shape[0], -1)


def evaluate_accuracy(data_iter, network):
    acc_sum, n = 0.0, 0
    for x_i, y_i in data_iter:
        acc_sum += (network(x_i).argmax(1) == y_i).float().sum().item()
        n += y_i.shape[0]
    return acc_sum / n


# 定义模型
inputs = 784
outputs = 10
hiddens = 256

net = nn.Sequential(
    FlattenLayer(),
    nn.Linear(inputs, hiddens),
    nn.ReLU(),
    nn.Linear(hiddens, outputs)
)

for params in net.parameters():
    init.normal_(params, 0, 0.01)

# 下载并读取数据集
mnist_train = datasets.FashionMNIST(r'.\Datasets', train=True, transform=transforms.ToTensor(), download=True)
mnist_test = datasets.FashionMNIST(r'.\Datasets', train=False, transform=transforms.ToTensor(), download=True)

# 生成数据
batch_size = 256
train_iter = dat.DataLoader(mnist_train, batch_size=batch_size, shuffle=True)
test_iter = dat.DataLoader(mnist_test, batch_size=batch_size, shuffle=False)

lr = 0.5
loss = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr)

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

