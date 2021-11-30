import torch
import math
from torch import nn
import numpy as np

if torch.cuda.is_available():
    device = torch.device('cuda:0')
else:
    device = torch.device('cpu')


def attention(q, k, v, mask=None):
    tmp = torch.matmul(q, k.transpose(1, 2)) / math.sqrt(q.shape[-1])
    if mask is not None:
        tmp = tmp.masked_fill(mask, -np.inf)
    tmp = nn.Softmax(-1)(tmp)
    return torch.matmul(tmp, v), tmp


class MultiHeadAttention(nn.Module):
    def __init__(self, head_n, in_feats, activation=None, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        assert in_feats % head_n == 0, f"in_features ({in_feats}) should be dividable by head_number ({head_n})"
        self.in_feats = in_feats
        self.head_n = head_n
        self.lin_q = nn.Linear(in_feats, in_feats)
        self.lin_k = nn.Linear(in_feats, in_feats)
        self.lin_v = nn.Linear(in_feats, in_feats)
        self.fc = nn.Linear(in_feats, in_feats)
        self.activation = activation
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(in_feats)

    def forward(self, q, k, v, mask=None):
        residual = q

        feat_head = self.in_feats // self.head_n
        batch, leng, _ = q.shape

        values, linears = [], [self.lin_q, self.lin_k, self.lin_v]
        for i, value in enumerate((q, k, v)):
            value = linears[i](value)
            if self.activation is not None:
                value = self.activation(value)
            value = value.reshape(batch, leng, self.head_n, feat_head)
            value = value.transpose(1, 2)
            value = value.reshape(batch * self.head_n, leng, feat_head)
            values.append(value)

        q, k, v = values
        if mask is not None:
            mask = mask.repeat(self.head_n, 1, 1)
        att_value, raw_att = attention(q, k, v, mask)

        att_value = att_value.reshape(self.head_n, batch, leng, feat_head)\
            .permute(1, 2, 0, 3).reshape(batch, leng, self.head_n * feat_head)

        att_value = self.fc(att_value)
        if self.activation is not None:
            att_value = self.activation(att_value)

        att_value = self.dropout(att_value)
        output = self.layer_norm(att_value + residual)

        return output, raw_att
