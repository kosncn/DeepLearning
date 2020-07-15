import time
import math
import random
import zipfile

import torch
from torch import nn, optim, cuda
from torch.nn import functional as F


class RNNModel(nn.Module):
    def __init__(self, rnn_layer, vocab_size):
        super().__init__()
        self.rnn = rnn_layer
        self.hiddden_size = rnn_layer.hidden_size * (2 if rnn_layer.bidirectional else 1)
        self.vocab_size = vocab_size
        self.dense = nn.Linear(self.hiddden_size, vocab_size)
        self.state = None

    def forward(self, inputs, state):  # inputs: (batch, seq_len)
        # 获取 one_hot 向量表示
        x = to_one_hot(inputs, self.vocab_size)  # x 是一个 list
        y, self.state = self.rnn(torch.stack(x).float(), state)  # torch.stack(x) 将列表 x 中所有 tensor 合并成一个 tensor
        output = self.dense(y.view(-1, y.shape[-1]))
        return output, self.state


# 相邻采样
def data_iter_consecutive(corpus_index, batch_size, num_steps, device=None):
    if device is None:
        device = torch.device('cuda' if cuda.is_available() else 'cpu')

    num_batch = len(corpus_index) // batch_size
    num_epoch = (num_batch - 1) // num_steps

    corpus_index = torch.tensor(corpus_index, dtype=torch.int64, device=device)
    corpus_index = corpus_index[:batch_size * num_batch].view(batch_size, num_batch)

    for i in range(num_epoch):
        i *= num_steps
        x = corpus_index[:, i:i + num_steps]
        y = corpus_index[:, i + 1: i + 1 + num_steps]
        yield x, y


def to_one_hot(x, n_class):
    return [F.one_hot(x[:, i], n_class) for i in range(x.shape[1])]


def grad_clipping(params, theta, device):
    norm = torch.tensor([0.0], device=device)
    for param in params:
        norm += (param.grad.data ** 2).sum()
    norm = norm.sqrt().item()
    if norm > theta:
        for param in params:
            param.grad.data *= (theta / norm)


def train_and_predict(model, device, corpus_index,
                      chars, index, num_epochs, num_steps, batch_size,
                      lr, clipping_theta, pred_period, pred_len, prefixes):
    loss = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr)
    model.to(device)
    state = None

    for epoch in range(num_epochs):
        los_sum, n, start = 0.0, 0, time.time()
        data_iter = data_iter_consecutive(corpus_index, batch_size, num_steps, device)
        for x, y in data_iter:
            if state is not None:
                state = state.detach_()

            output, state = model(x, state)  # output 的形状为 (num_steps * batch_size, vocab_size)

            # y 的形状为 (batch_size, num_steps)，转置后将变成长度为 batch_size * num_steps 的向量，这样跟输出的行一一对应
            y = torch.transpose(y, 0, 1).contiguous().view(-1)
            los = loss(output, y.long())

            optimizer.zero_grad()
            los.backward()
            grad_clipping(model.parameters(), clipping_theta, device)
            optimizer.step()
            los_sum += los.item() * y.shape[0]
            n += y.shape[0]

        if (epoch + 1) % pred_period == 0:
            print(f'epoch: {epoch + 1}, '
                  f'perplexity: {math.exp(los_sum / n)}, '
                  f'time: {time.time() - start}')
            for prefix in prefixes:
                print('-', predict(prefix, pred_len, model, chars, index, device))


def predict(prefix, num_chars, model, chars, index, device):
    state = None
    output = [index[prefix[0]]]  # output 将记录 prefix 加上输出
    for t in range(num_chars + len(prefix) - 1):
        x = torch.tensor([output[-1]], device=device).view(1, 1)
        if state is not None:
            state = state.to(device)
        y, state = model(x, state)
        if t < len(prefix) - 1:
            output.append(index[prefix[t + 1]])
        else:
            output.append(int(y.argmax(dim=1).item()))
    return ''.join([chars[i] for i in output])


device = torch.device('cuda' if cuda.is_available() else 'cpu')

# 读取数据集
with zipfile.ZipFile('Datasets/jaychou_lyrics.txt.zip') as zin:
    with zin.open('jaychou_lyrics.txt') as f:
        corpus_chars = f.read().decode('utf-8')

corpus_chars = corpus_chars.replace('\n', ' ').replace('\r', ' ')
corpus_chars = corpus_chars[:10000]

# 建立字符索引
chars = sorted(list(set(corpus_chars)))
index = dict([(char, i) for i, char in enumerate(chars)])
vocab_size = len(index)

corpus_index = [index[char] for char in corpus_chars]

# 初始化模型参数
num_hiddens = 256
rnn_layer = nn.RNN(input_size=vocab_size, hidden_size=num_hiddens)

num_steps = 35
# batch_size = 2
# state = None
# x = torch.rand(num_steps, batch_size, vocab_size)
# y, state_new = rnn_layer(x, state)
# print(y.shape)
# print(state_new.shape)
# print(state_new[0].shape)

# 测试 predict 函数
model = RNNModel(rnn_layer, vocab_size).to(device)
# print(predict('分开', 10, model, chars, index, device))

# 初始化超参数
num_epochs, batch_size, lr, clipping_theta = 250, 32, 1e-3, 1e-2
pred_period, pred_len, prefixes = 50, 50, ['分开', '不分开']

# 训练模型并创作歌词
train_and_predict(model, device, corpus_index, chars,
                  index, num_epochs, num_steps, batch_size,
                  lr, clipping_theta, pred_period, pred_len, prefixes)
