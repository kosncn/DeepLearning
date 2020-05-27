import torch
from torch import nn
from torch.nn import init


class MyModule(nn.Module):
    def __init__(self):
        super(MyModule, self).__init__()
        self.weight1 = nn.Parameter(torch.rand(20, 20))
        self.weight2 = torch.rand(20, 20)


def init_weight_(tensor):
    with torch.no_grad():
        tensor.uniform_(-10, 10)
        tensor *= (tensor.abs() >= 5).float()


net = nn.Sequential(nn.Linear(5, 3), nn.ReLU(), nn.Linear(3, 1))  # pytorch已进行默认初始化
x = torch.rand(3, 5)
y = net(x).sum()
# print(net)
# print(y)

# 访问模型参数
# print(net.named_parameters())
# for name, param in net.named_parameters():
#     print(name, param.size())

# print(net[0])
# for name, param in net[0].named_parameters():
#     print(name, param.size(), type(param))

# n = MyModule()
# for name, param in n.named_parameters():
#     print(name)  # weight1

# weight_0 = list(net[0].parameters())[0]
# print(weight_0.data)
# print(weight_0.grad)  # 反向传播前梯度为None
# y.backward()
# print(weight_0.grad)

# 初始化模型参数
# for name, param in net.named_parameters():
#     if 'weight' in name:
#         init.normal_(param, 0, 0.01)
#         # print(name, param.data)
#
# for name, param in net.named_parameters():
#     if 'bias' in name:
#         init.constant_(param, 0)
#         # print(name, param.data)

# 使用自定义方法初始化参数
# for name, param in net.named_parameters():
#     if 'weight' in name:
#         init_weight_(param)
#         print(name, param.data)

# for name, param in net.named_parameters():
#     if 'bias' in name:
#         param.data += 1
#         print(name, param.data)

# 共享模型参数
linear = nn.Linear(1, 1, False)
net = nn.Sequential(linear, linear)
print(net)
for name, param in net.named_parameters():
    init.constant_(param, 3)
    print(name, param.data)
print(id(net[0]) == id(net[1]))  # True
print(id(net[0].weight) == id(net[1].weight))  # True

x = torch.ones(1, 1)
y = net(x).sum()
print(y)
y.backward()
print(net[0].weight.grad)
