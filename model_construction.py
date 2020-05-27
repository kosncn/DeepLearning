import torch
from torch import nn
from torch.nn import functional as F

from collections import OrderedDict


class MLP(nn.Module):
    # 声明含模型参数的层
    def __init__(self, ):
        super(MLP, self).__init__()
        self.hidden = nn.Linear(784, 256)  # 隐藏层
        self.active = nn.ReLU()
        self.output = nn.Linear(256, 10)  # 输出层

    # 定义模型的前向计算
    def forward(self, x):
        a = self.active(self.hidden(x))
        return self.output(a)


class MySequential(nn.Module):
    def __init__(self, *args):
        super(MySequential, self).__init__()
        if len(args) == 1 and isinstance(args[0], OrderedDict):  # 如果传入的参数是一个OrderedDict
            for name, module in args[0].items():
                self.add_module(name, module)  # self.add_module()方法会将module添加进self._modules(一个OrderedDict)
        else:  # 传入的是一些module
            for index, module in enumerate(args):
                self.add_module(str(index), module)

    def forward(self, input):
        # self._modules返回一个OrderedDict，将保证按照成员添加顺序遍历成员
        for module in self._modules.values():
            input = module(input)
        return input


class MyModuleList(nn.Module):
    def __init__(self):
        super(MyModuleList, self).__init__()
        self.linears = nn.ModuleList([nn.Linear(10, 10) for _ in range(10)])

    def forward(self, x):
        for i, j in enumerate(self.linears):
            x = self.linears[i // 2](x) + j(x)
        return x


class Module_ModuleList(nn.Module):
    def __init__(self):
        super(Module_ModuleList, self).__init__()
        self.linears = nn.ModuleList([nn.Linear(10, 10)])


class Module_List(nn.Module):
    def __init__(self):
        super(Module_List, self).__init__()
        self.linears = [nn.Linear(10, 10)]


class FancyMLP(nn.Module):
    def __init__(self):
        super(FancyMLP, self).__init__()
        self.weight = torch.rand(20, 20, requires_grad=False)  # 不可训练参数（常数参数）
        self.linear = nn.Linear(20, 20)

    def forward(self, x):
        x = self.linear(x)
        x = F.relu(torch.mm(x, self.weight.data) + 1)
        x = self.linear(x)  # 复用全连接层，等价于两个全连接层共享参数
        while x.norm().item() > 1:
            x /= 2
        if x.norm().item() < 0.8:
            x *= 10
        return x.sum()


class NestMLP(nn.Module):
    def __init__(self):
        super(NestMLP, self).__init__()
        self.net = nn.Sequential(nn.Linear(50, 30), nn.ReLU())

    def forward(self, x):
        return self.net(x)


# 测试MLP类
# net = MLP()
# x = torch.rand(3, 784)
# print(net)
# print(net(x))

# 测试MySequential类
# net = MySequential(
#     nn.Linear(784, 256),
#     nn.ReLU(),
#     nn.Linear(256, 10)
# )
# x = torch.rand(3, 784)
# print(net)
# print(net(x))

# ModuleList类
# net = nn.ModuleList([nn.Linear(784, 256), nn.ReLU()])
# net.append(nn.Linear(256, 10))  # 支持类似List的append操作
# print(net[-1])  # 支持类似List的索引操作
# print(net)
# # print(net(torch.rand(3, 784)))  # 会报错：TypeError: forward() takes 1 positional argument but 2 were given；原因：nn.ModuleList内部未实现forward()方法

# 测试MyModuleList类
# net = MyModuleList()
# x = torch.rand(3, 10)
# print(net)
# print(net(x))

# ModuleList与Python中List的区别：加入到ModuleList里面的所有模块的参数会被自动添加到整个网络中
# net1 = Module_ModuleList()
# net2 = Module_List()
# print('net1:')
# for p in net1.parameters():
#     print(p.size())
# print('net2:')
# for p in net2.parameters():
#     print(p.size())

# ModuleDict类
# net = nn.ModuleDict({
#     'linear': nn.Linear(784, 256),
#     'active': nn.ReLU()
# })
# net['output'] = nn.Linear(256, 10)  # 支持类似字典的添加操作
# print(net['linear'])  # 支持类似字典的访问操作
# print(net.output)
# print(net)
# # print(net(torch.rand(3, 784)))  # 会报错：TypeError: forward() takes 1 positional argument but 2 were given；原因：nn.ModuleDict内部未实现forward()方法
# # ModuleDict与Python中Dict的区别：加入到ModuleDict里面的所有模块的参数会被自动添加到整个网络中

# 测试FancyMLP类
# net = FancyMLP()
# x = torch.rand(3, 20)
# print(net)
# print(net(x))

# 测试NestMLP类
# net = nn.Sequential(
#     NestMLP(),
#     nn.Linear(30, 20),
#     FancyMLP()
# )
# x = torch.rand(3, 50)
# print(net)
# print(net(x))

