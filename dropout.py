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


# def net(x_n, is_training=True):
#     h1 = (torch.matmul(x_n.view(-1, inputs), w1) + b1).relu()
#     if is_training:
#         h1 = dropout(h1, drop_prob1)
#     h2 = (torch.matmul(h1, w2) + b2).relu()
#     if is_training:
#         h2 = dropout(h2, drop_prob2)
#     return torch.matmul(h2, w3) + b3


def dropout(x_d, drop_prob):
    assert 0 <= drop_prob <= 1
    x_d = x_d.float()
    keep_prob = 1 - drop_prob
    if keep_prob == 0:
        return torch.zeros_like(x_d)
    mask = (torch.rand(x_d.size()) < keep_prob).float()
    return mask / keep_prob * x_d


# 定义更新函数
def sgd(para, l_r):
    for p in para:
        p.data -= l_r * p.grad


def evaluate_accuracy(data_iter, network):
    acc_sum, n = 0.0, 0
    for x_i, y_i in data_iter:
        if isinstance(net, nn.Module):
            net.eval()  # 改为评估模式，会关闭dropout
            acc_sum += (network(x_i).argmax(1) == y_i).float().sum().item()
            net.train()  # 改为训练模式
        else:
            acc_sum += (network(x_i, False).argmax(1) == y_i).float().sum().item()
        n += y_i.shape[0]
    return acc_sum / n


# 测试dropout
# x_t = torch.arange(15).view(3, 5)
# print(dropout(x_t, 0.0))
# print(dropout(x_t, 0.5))
# print(dropout(x_t, 1.0))

# 定义模型参数
inputs, outputs, hiddens1, hiddens2 = 784, 10, 256, 256

w1 = torch.normal(0, 0.01, (inputs, hiddens1), requires_grad=True)
b1 = torch.zeros(hiddens1, requires_grad=True)
w2 = torch.normal(0, 0.01, (hiddens1, hiddens2), requires_grad=True)
b2 = torch.zeros(hiddens2, requires_grad=True)
w3 = torch.normal(0, 0.01, (hiddens2, outputs), requires_grad=True)
b3 = torch.zeros(outputs, requires_grad=True)

params = [w1, b1, w2, b2, w3, b3]

# 定义模型
drop_prob1, drop_prob2 = 0.2, 0.5

# 训练和测试模型
batch_size, epochs, lr = 256, 5, 0.5
loss = nn.CrossEntropyLoss()

mnist_train = datasets.FashionMNIST(r'.\Datasets', train=True, transform=transforms.ToTensor(), download=True)
mnist_test = datasets.FashionMNIST(r'.\Datasets', train=False, transform=transforms.ToTensor(), download=True)
train_iter = dat.DataLoader(mnist_train, batch_size=batch_size, shuffle=True)
test_iter = dat.DataLoader(mnist_test, batch_size=batch_size, shuffle=False)

# for epoch in range(epochs):
#     train_loss, train_acc, count = 0.0, 0.0, 0
#     for x, y in train_iter:
#         y_pred = net(x)
#         los = loss(y_pred, y).sum()
#         los.backward()
#         sgd([w1, b1, w2, b2, w3, b3], lr)
#         for p in params:
#             p.grad.data.zero_()
#         train_loss += los.item()
#         train_acc += (y_pred.argmax(1) == y).float().sum().item()
#         count += y.shape[0]
#     test_acc = evaluate_accuracy(test_iter, net)
#     print(f'epoch: {epoch + 1}, loss: {train_loss / count:.5f}, train acc: {train_acc / count:.5f}, test acc: {test_acc:.5f}')

# ============================== 简洁实现 ============================== #
net = nn.Sequential(
    FlattenLayer(),
    nn.Linear(inputs, hiddens1),
    nn.ReLU(),
    nn.Dropout(drop_prob1),
    nn.Linear(hiddens1, hiddens2),
    nn.ReLU(),
    nn.Dropout(drop_prob2),
    nn.Linear(hiddens2, outputs)
)

for p in net.parameters():
    init.normal_(p, 0, 0.01)

optimizer = optim.SGD(net.parameters(), lr)

for epoch in range(epochs):
    train_loss, train_acc, count = 0.0, 0.0, 0
    for x, y in train_iter:
        y_pred = net(x)
        los = loss(y_pred, y).sum()
        los.backward()
        optimizer.step()
        optimizer.zero_grad()
        train_loss += los.item()
        train_acc += (y_pred.argmax(1) == y).float().sum().item()
        count += y.shape[0]
    test_acc = evaluate_accuracy(test_iter, net)
    print(f'epoch: {epoch + 1}, loss: {train_loss / count:.5f}, train acc: {train_acc / count:.5f}, test acc: {test_acc:.5f}')
