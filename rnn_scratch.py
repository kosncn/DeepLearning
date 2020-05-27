import torch
from torch import nn, optim, cuda
from torch.nn import functional as F

import random
import zipfile


# 随机采样
def data_iter_random(corpus_index, batch_size, num_steps, device=None):
    # 返回从pos开始的长为num_steps的序列
    def data(pos):
        return corpus_index[pos: pos + num_steps]

    if device is None:
        device = torch.device('cuda' if cuda.is_available() else 'cpu')

    example_nums = (len(corpus_index) - 1) // num_steps
    example_index = list(range(example_nums))
    random.shuffle(example_index)

    num_epochs = example_nums // batch_size
    print(num_epochs)

    for i in range(num_epochs):
        i *= batch_size
        batch_index = example_index[i: i + batch_size]
        x = [data(j * num_steps) for j in batch_index]
        y = [data(j * (num_steps + 1)) for j in batch_index]
        yield torch.tensor(x, dtype=torch.float, device=device), torch.tensor(y, dtype=torch.float, device=device)


# 相邻采样
def data_iter_consecutive(corpus_index, batch_size, num_steps, device=None):
    if device is None:
        device = torch.device('cuda' if cuda.is_available() else 'cpu')

    batch_len = len(corpus_index) // batch_size
    num_epoch = (batch_len - 1) // num_steps

    corpus_index = torch.tensor(corpus_index, dtype=torch.float, device=device)
    corpus_index = corpus_index[:batch_size * batch_len].view(batch_size, batch_len)

    for i in range(num_epoch):
        i *= num_steps
        x = corpus_index[:, i:i + num_steps]
        y = corpus_index[:, i + 1: i + 1 + num_steps]
        yield x, y


def to_one_hot(x, n_class):
    return [F.one_hot(x[:, i], n_class) for i in range(x.shape[1])]


def get_params():
    # 隐藏层参数
    w_xh = nn.Parameter(torch.normal(0, 0.01, (num_inputs, num_hiddens), device=device, requires_grad=True))
    w_hh = nn.Parameter(torch.normal(0, 0.01, (num_hiddens, num_hiddens), device=device, requires_grad=True))
    b_h = nn.Parameter(torch.zeros(num_hiddens, device=device, requires_grad=True))

    # 输出层参数
    w_hq = nn.Parameter(torch.normal(0, 0.01, (num_hiddens, num_outputs), device=device, requires_grad=True))
    b_q = nn.Parameter(torch.zeros(num_outputs, device=device, requires_grad=True))

    return nn.ParameterList([w_xh, w_hh, b_h, w_hq, b_q])


def init_rnn_state(batch_size, num_hiddens, device):
    return torch.zeros((batch_size, num_hiddens), device=device)


def rnn(inputs, state, params):
    h = state
    outputs = []
    w_xh, w_hh, b_h, w_hq, b_q = params
    for x in inputs:
        h = torch.tanh(torch.matmul(x, w_xh) + torch.matmul(h, w_hh) + b_h)
        y = torch.matmul(h, w_hq) + b_q
        outputs.append(y)
    return outputs, h


device = torch.device('cuda' if cuda.is_available() else 'cpu')

# x, w_xh = torch.randn(3, 1), torch.randn(1, 4)
# h, w_hh = torch.randn(3, 4), torch.randn(4, 4)

# print(torch.matmul(x, w_xh) + torch.matmul(h, w_hh))
# print(torch.matmul(torch.cat((x, h), dim=1), torch.cat((w_xh, w_hh), dim=0)))

# 读取数据集
with zipfile.ZipFile('Datasets/jaychou_lyrics.txt.zip') as zin:
    with zin.open('jaychou_lyrics.txt') as f:
        corpus_chars = f.read().decode('utf-8')
# print(corpus_chars[:55])

corpus_chars = corpus_chars.replace('\n', ' ').replace('\r', ' ')
corpus_chars = corpus_chars[:10000]

# 建立字符索引
chars = sorted(list(set(corpus_chars)))
index = dict([(char, i) for i, char in enumerate(chars)])
vocab_size = len(index)
# print(vocab_size)

corpus_index = [index[char] for char in corpus_chars]
# print(f'chars: {corpus_chars[:30]}')
# print(f'index: {corpus_index[:30]}')

# 测试data_iter_random函数
# sequence = list(range(30))
# for x, y in data_iter_random(sequence, 2, 6):
#     print(f'x: {x}')
#     print(f'y: {y}')

# 测试data_iter_consecutive函数
# sequence = list(range(30))
# for x, y in data_iter_consecutive(sequence, 2, 6):
#     print(f'x: {x}')
#     print(f'y: {y}')

# 测试F.one_hot函数
# x = torch.tensor([0, 1, 2])
# print(F.one_hot(x, vocab_size))

# 测试to_one_hot函数
# x = torch.arange(15).view(3, 5)
# inputs = to_one_hot(x, vocab_size)
# print(len(inputs))
# print(inputs[0].shape)

# 初始化模型参数
num_inputs, num_hiddens, num_outputs = vocab_size, 256, vocab_size

# 测试cnn函数
x = torch.arange(15).view(3, 5)
state = init_rnn_state(x.shape[0], num_hiddens, device)
inputs = to_one_hot(x.to(device), vocab_size)
params = get_params()
outputs, state_new = rnn(inputs, state, params)
print(len(outputs))
print(outputs[0].shape)
print(state_new.shape)



