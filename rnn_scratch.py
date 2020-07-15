import time
import math
import random
import zipfile

import torch
from torch import nn, cuda
from torch.nn import functional as F


# 随机采样
def data_iter_random(corpus_index, batch_size, num_steps, device=None):
    # 返回从pos开始的长为num_steps的序列
    def data(pos):
        return corpus_index[pos: pos + num_steps]

    if device is None:
        device = torch.device('cuda' if cuda.is_available() else 'cpu')

    num_examples = (len(corpus_index) - 1) // num_steps  # 减一是因为输出的索引y是相应输入索引x加一
    idx_examples = list(range(num_examples))
    random.shuffle(idx_examples)

    num_epochs = num_examples // batch_size

    for i in range(num_epochs):
        i *= batch_size
        batch_index = idx_examples[i: i + batch_size]
        x = [data(j * num_steps) for j in batch_index]
        y = [data(j * num_steps + 1) for j in batch_index]
        yield torch.tensor(x, dtype=torch.int64, device=device), torch.tensor(y, dtype=torch.int64, device=device)


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
    # print(x)
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


def init_state(batch_size, num_hiddens, device):
    return torch.zeros((batch_size, num_hiddens), device=device)


def rnn(inputs, state, params):
    # inputs 和 outputs 均为 num_steps 个形状为 (batch_size, vocab_size) 的矩阵
    h = state
    outputs = []
    w_xh, w_hh, b_h, w_hq, b_q = params
    for x in inputs:
        h = torch.tanh(torch.matmul(x.float(), w_xh) + torch.matmul(h, w_hh) + b_h)
        y = torch.matmul(h, w_hq) + b_q
        outputs.append(y)
    return outputs, h


def grad_clipping(params, theta, device):
    norm = torch.tensor([0.0], device=device)
    for param in params:
        norm += (param.grad.data ** 2).sum()
    norm = norm.sqrt().item()
    if norm > theta:
        for param in params:
            param.grad.data *= (theta / norm)


# 定义优化算法
def sgd(params, l_r, b_s):
    for param in params:
        param.data -= l_r * param.grad / b_s


def train_and_predict(rnn, get_params, init_state, num_hiddens, vocab_size, device,
                      corpus_index, chars, index, is_random_iter, num_epochs, num_steps,
                      batch_size, lr, clipping_theta, pred_period, pred_len, prefixes):
    data_iter_fn = data_iter_random if is_random_iter else data_iter_consecutive
    params = get_params()
    loss = nn.CrossEntropyLoss()

    for epoch in range(num_epochs):
        if not is_random_iter:  # 如果使用相邻采样，则在 epoch 开始时初始化隐藏状态
            state = init_state(batch_size, num_hiddens, device)
        loss_sum, n, start = 0.0, 0, time.time()
        data_iter = data_iter_fn(corpus_index, batch_size, num_steps, device)
        for x, y in data_iter:
            if is_random_iter:  # 如果使用随机采样，则在每个小批量更新前初始化隐藏状态
                state = init_state(batch_size, num_hiddens, device)
            else:
                # 如果使用相邻采样，需要使用 detach 函数将隐藏状态从计算图中分离出来，
                # 这是为了是模型参数的梯度计算之依赖一次迭代读取的小批量序列（防止梯度计算开销太大）
                if state.grad_fn is not None:
                    state = state.detach_()

            inputs = to_one_hot(x, vocab_size)
            # outputs 有 num_steps 个形状为 (batch_size, vocab_size) 的矩阵
            outputs, state = rnn(inputs, state, params)
            # outputs 拼接之后的形状为 (num_steps * batch_size, vocab_size)
            outputs = torch.cat(outputs, dim=0)
            # y 的形状为 (batch_size, num_steps)，转置后变为长度为 batch_size * num_steps 的向量，
            # 这样跟输出的行一一对应
            y = torch.transpose(y, 0, 1).contiguous().view(-1)  # contiguous() 函数作用：开辟一块新的内存来存放变换之后的数据
            # 使用交叉熵损失函数计算平均分类误差
            los = loss(outputs, y)

            # 梯度清零
            if params[0].grad is not None:
                for param in params:
                    param.grad.data.zero_()
            los.backward()
            # 裁剪梯度
            grad_clipping(params, clipping_theta, device)
            sgd(params, lr, 1)  # 因为误差已经取过均值，梯度不需要再做平均
            loss_sum += los.item() * y.shape[0]
            n += y.shape[0]
        if (epoch + 1) % pred_period == 0:
            print(f'epoch: {epoch + 1}, '
                  f'perplexity: {math.exp(loss_sum / n)}, '
                  f'time: {time.time() - start}')
            for prefix in prefixes:
                print('-', predict(prefix, pred_len, rnn, params,
                                   init_state, num_hiddens, chars, index,
                                   vocab_size, device))


def predict(prefix, num_chars, rnn, params, init_state, num_hiddens, chars, index, vocab_size, device):
    state = init_state(1, num_hiddens, device)
    outputs = [index[prefix[0]]]
    for t in range(num_chars + len(prefix) - 1):
        # 将上一时间步的输出作为当前时间步的输入
        x = to_one_hot(torch.tensor([[outputs[-1]]], dtype=torch.int64, device=device), vocab_size)
        # 计算输出和更新隐藏状态
        y, state = rnn(x, state, params)
        # 下一个时间步的输入是 prefix 里的字符或者当前的最佳预测字符
        if t < len(prefix) - 1:
            outputs.append(index[prefix[t + 1]])
        else:
            outputs.append(int(y[0].argmax(dim=1).item()))
    return ''.join([chars[i] for i in outputs])


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
# print(len(corpus_chars))  # 63282

corpus_chars = corpus_chars.replace('\n', ' ').replace('\r', ' ')
corpus_chars = corpus_chars[:10000]

# 建立字符索引
chars = sorted(list(set(corpus_chars)))
index = dict([(char, i) for i, char in enumerate(chars)])
vocab_size = len(index)
# print(vocab_size)  # 1027

corpus_index = [index[char] for char in corpus_chars]
# print(f'chars: {corpus_chars[:35]}')
# print(f'index: {corpus_index[:35]}')

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
# print(inputs)
# print(len(inputs))
# print(inputs[0].shape)

# 初始化模型参数
num_inputs, num_hiddens, num_outputs = vocab_size, 256, vocab_size

# 测试cnn函数
# x = torch.arange(15).view(3, 5)
# state = init_state(x.shape[0], num_hiddens, device)
# inputs = to_one_hot(x.to(device), vocab_size)
params = get_params()
# outputs, state_new = rnn(inputs, state, params)
# print(len(outputs))
# print(outputs[0].shape)
# print(state_new.shape)

# 测试predict函数
# print(predict('分开', 10, rnn, params, init_state, num_hiddens, chars, index, vocab_size, device))

# 初始化超参数
num_epochs, num_steps, batch_size, lr, clipping_theta = 250, 35, 32, 1e2, 1e-2
pred_period, pred_len, prefixes = 50, 50, ['分开', '不分开']

# 训练模型并创作歌词
train_and_predict(rnn, get_params, init_state, num_hiddens, vocab_size, device,
                  corpus_index, chars, index, True, num_epochs, num_steps,
                  batch_size, lr, clipping_theta, pred_period, pred_len, prefixes)

