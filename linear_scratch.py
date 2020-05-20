import torch

import random

from matplotlib import pyplot as plt


# 小批量数据生成函数
def data_iter(b_s, x_t, y_t):
    count = len(x_t)
    index = list(range(count))
    random.shuffle(index)
    for i in range(0, count, b_s):
        j = torch.tensor(index[i: min(i + b_s, count)])
        yield x_t.index_select(0, j), y_t.index_select(0, j)


# 定义模型
def linear(x_m, w_m, b_m):
    return torch.mm(x_m, w_m) + b_m


# 定义损失函数
def squared_loss(y_h, y_t):
    return (y_h - y_t.view(-1, 1)) ** 2 / 2


# 定义优化算法
def sgd(params, l_r, b_s):
    for param in params:
        param.data -= l_r * param.grad / b_s


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

# 绘图观察生成的第二个特征与标签间的线性关系
# fig = plt.figure(figsize=(5, 3))
# plt.scatter(x[:, 1].numpy(), y.numpy(), 1)
# plt.show()

# 初始化模型参数
w = torch.normal(0, 0.01, (features, 1), requires_grad=True)
b = torch.zeros(1, requires_grad=True)
print(w)
print(b)

# 训练模型
lr = 0.03
epochs = 5
batch_size = 10
for epoch in range(epochs):
    for x_b, y_b in data_iter(batch_size, x, y):
        loss = squared_loss(linear(x_b, w, b), y_b).sum()
        loss.backward()
        sgd([w, b], lr, batch_size)
        w.grad.data.zero_()  # 清零梯度
        b.grad.data.zero_()  # 清零梯度
    train_loss = squared_loss(linear(x, w, b), y).sum()  # 训练集误差
    print(f'epoch: {epoch + 1}, loss: {train_loss.item()}')

# 打印真实参数与学到参数
print(f'true_w: {true_w}, train_w: {w}')
print(f'true_b: {true_b}, train_b: {b}')
