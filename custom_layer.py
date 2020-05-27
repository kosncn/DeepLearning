import torch
from torch import nn


class CenteredLayer(nn.Module):
    def __init__(self):
        super(CenteredLayer, self).__init__()

    def forward(self, x):
        return x - x.mean()


class MyListDense(nn.Module):
    def __init__(self):
        super(MyListDense, self).__init__()
        self.params = nn.ParameterList([nn.Parameter(torch.randn(3, 3)) for _ in range(2)])
        self.params.append(nn.Parameter(torch.randn(3, 1)))

    def forward(self, x):
        for i in range(len(self.params)):
            x = torch.mm(x, self.params[i])
        return x


class MyDictDense(nn.Module):
    def __init__(self):
        super(MyDictDense, self).__init__()
        self.params = nn.ParameterDict({
            'linear1': nn.Parameter(torch.randn(3, 3)),
            'linear2': nn.Parameter(torch.randn(3, 1))
        })
        self.params.update({'linear3': nn.Parameter(torch.randn(3, 2))})  # 新增

    def forward(self, x, choice='linear1'):
        return torch.mm(x, self.params[choice])


# 不含模型参数的自定义层
# layer = CenteredLayer()
# print(layer(torch.tensor([1, 2, 3, 4, 5], dtype=torch.float)))

# net = nn.Sequential(nn.Linear(8, 128), CenteredLayer())
# y = net(torch.rand(4, 8))
# print(y.mean().item())


# 含模型参数的自定义层
# net = MyListDense()
# print(net)

# net = MyDictDense()
x = torch.ones(1, 3)
# print(net)
# print(net(x, 'linear1'))
# print(net(x, 'linear2'))
# print(net(x, 'linear3'))

net = nn.Sequential(
    MyDictDense(),
    MyListDense()
)
print(net)
print(net(x))
