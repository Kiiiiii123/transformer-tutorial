import math
from turtle import forward

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)


# d_model = 512
# vocab = 1000
# x = Variable(torch.LongTensor([[100, 2, 421, 508], [491, 998, 1, 221]]))
# emb = Embeddings(d_model, vocab)
# embr = emb(x)
# print('embr:', embr)
# print(embr.shape)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()

        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arrange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arrange(0, d_model, 2) * -(math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + Variable(self.pe[:, : x.size(1)], require_grad=False)
        return self.dropout(x)


# d_model = 512
# dropout = 0.1
# max_len = 60
# x = embr
# pe = PositionalEncoding(d_model, dropout, max_len)
# pe_result = pe(x)
# print('pe_result:', pe_result)


def subsequent_mask(size):
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype("uint8")
    return torch.from_numpy(1 - subsequent_mask)


# Attention Machanism
def attention(query, key, value, mask=None, dropout=None):
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e-9)

    p_attn = F.softmax(scores, dim=-1)

    if dropout is not None:
        p_attn = dropout(p_attn)

    return torch.matmul(p_attn, value), p_attn


# query = key = value = pe_result
# mask = torch.Variable(torch.zeros(2, 4, 4))
# attn, p_attn = attention(query, key, value, mask=mask)
# print('attn', attn)
# print(attn.shape)
# print('p_attn', p_attn)
# print(p_attn.shape)


# Multi-head Attention Machanism
import copy  # for deep copy


def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class MultiHeadedAttention(nn.Module):
    def __init__(self, head, embedding_dim, dropout=0.1):
        super(MultiHeadedAttention, self).__init__()

        assert embedding_dim % head == 0

        self.d_k = embedding_dim // head
        self.head = head
        self.linears = clones(nn.Linear(embedding_dim, embedding_dim), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        if mask is not None:
            mask = mask.unsqueeze(1)

        batch_size = query.size(0)
        query, key, value = [
            model(x).view(batch_size, -1, self.head, self.d_k).transpose(1, 2)
            for model, x in zip(self.linears, (query, key, value))
        ]
        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.d_k * self.head)

        return self.linears[-1](x)


# head = 8
# embedding_dim =512
# dropout = 0.2
# query = key = value = pe_result
# mask = Variable(torch.zeros(8, 4, 4))
# mha = MultiHeadedAttention(head, embedding_dim, dropout)
# mha_result = mha(query, key, value, mask)
# print(mha_result)
# print(mha_result.shape)


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()

        self.w1 = nn.Linear(d_model, d_ff)
        self.w2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w2(self.dropout(F.relu(self.w1(x))))


# d_model = 512
# d_ff = 64
# dropout = 0.2
# ff = PositionwiseFeedForward(d_model, d_ff, dropout)
# ff_result = ff(x)
# print(ff_result)


class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()

        self.a2 = nn.Parameter(torch.ones(features))
        self.b2 = nn.Parameter(torch.zeros(features))

        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keep_dim=True)
        std = x.std(-1, keep_dim=True)
        return self.a2 * (x - mean) / (std + self.eps) + self.b2


