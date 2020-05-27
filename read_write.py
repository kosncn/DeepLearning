import torch
from torch import nn
from torch import optim


class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.hidden = nn.Linear(3, 2)
        self.active = nn.ReLU()
        self.output = nn.Linear(2, 1)

    def forward(self, x):
        active = self.active(self.hidden(x))
        return self.output(active)


# 读写Tensor
# 存储并读取一个Tensor
# x = torch.ones(3)
# torch.save(x, 'x.pt')
#
# x2 = torch.load('x.pt')
# print(x2)

# 存储并读取一个Tensor列表
# y = torch.zeros(3)
# torch.save([x, y], 'xy.pt')
# xy_list = torch.load('xy.pt')
# print(xy_list)

# 存储并读取一个字符串映射到Tensor的字典
# torch.save({'x': x, 'y': y}, 'xy_dict.pt')
# xy_dict = torch.load('xy_dict.pt')
# print(xy_dict)

# 读写模型
net = MLP()
# print(net.state_dict())

# optimizer = optim.SGD(net.parameters(), 0.001, 0.9)
# print(optimizer.state_dict())

# 保存和读取state_dict
# torch.save(net.state_dict(), 'net.pt')
#
# net.load_state_dict(torch.load('net.pt'))
# print(net.state_dict())

x = torch.randn(3, 3)
y = net(x)
torch.save(net.state_dict(), 'net.pt')

net2 = MLP()
net2.load_state_dict(torch.load('net.pt'))
y2 = net2(x)
print(y2 == y)


