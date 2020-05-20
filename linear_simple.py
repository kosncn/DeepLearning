import torch

from torch import nn
from torch import optim
from torch.nn import init
from torch.utils import data as dat

from collections import OrderedDict


# 生成数据集
features = 2
samples = 1000
true_w = torch.tensor([2, -3.4]).view(2, 1)
true_b = torch.tensor([4.2]).view(1, 1)
# print(true_w)
# print(true_b)

x = torch.randn(samples, features)
y = torch.mm(x, true_w) + true_b
y += torch.normal(0, 0.01, y.size())
# print(x)
# print(y)

# 读取数据
batch_size = 10
data_set = dat.TensorDataset(x, y)  # 组合训练数据的x和y
data_iter = dat.DataLoader(data_set, batch_size, True)
# for i, j in data_iter:
#     print(i, j)
#     break


# 定义模型（方式一）
# class LinearNet(nn.Module):
#     def __init__(self, n_feature):
#         super().__init__()
#         self.linear = nn.Linear(n_feature, 1)
#
#     # 定义前向传播
#     def forward(self, x_t):
#         y_t = self.linear(x_t)
#         return y_t
#
#
# net = LinearNet(features)

# 定义模型（方式二）

# 写法一：
net = nn.Sequential(
    nn.Linear(features, 1)
    # 此处可以继续传入其他层

    # OrderedDict([
    #     ('linear', nn.Linear(features, 1))
    #     # 此处可以继续传入其他层
    # ])
)

# 写法二：
# net = nn.Sequential()
# net.add_module('linear', nn.Linear(features, 1))
# # net.add_module() ...

# print(net)
# print(net[0])

# 可通过 net.parameters() 查看模型所有学习参数，此函数返回一个生成器
# for param in net.parameters():
#     print(param)

# 初始化模型参数
init.normal_(net[0].weight, 0, 0.01)
init.constant_(net[0].bias, 0)

# 定义损失函数
loss = nn.MSELoss()

# 定义优化算法
optimizer = optim.SGD(net.parameters(), 0.03)
# print(optimizer)

# 训练模型
epochs = 5
for epoch in range(epochs):
    for x_b, y_b in data_iter:
        los = loss(net(x_b), y_b.view(-1, 1))
        los.backward()
        optimizer.step()
        optimizer.zero_grad()  # 梯度清零
    train_loss = loss(net(x), y.view(-1, 1))  # 训练集误差
    print(f'epoch: {epoch + 1}, loss: {train_loss.item()}')

# 打印真实参数与学到参数
dense = net[0]
print(f'true_w: {true_w}, train_w: {dense.weight}')
print(f'true_b: {true_b}, train_b: {dense.bias}')
