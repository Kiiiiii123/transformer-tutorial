import math
from turtle import forward

import numpy as np
import torch
import torch.nn as nn
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
