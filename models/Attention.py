import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import math
from math import sqrt
import os


class AutoCorrelation_Attention(nn.Module):
    def __init__(self, query_dim, key_dim, value_dim, num_heads: int=2, scale: int=1, dropout: float=0.1):
        super(AutoCorrelation_Attention, self).__init__()

        self.head_dim = query_dim // num_heads
        self.scale = scale
        self.num_heads = num_heads

        self.query_projection = nn.Linear(query_dim, num_heads * self.head_dim)
        self.key_projection = nn.Linear(key_dim, num_heads * self.head_dim)
        self.value_projection = nn.Linear(value_dim, num_heads * self.head_dim)
        self.out_projection = nn.Linear(num_heads*self.head_dim, query_dim)

        self.scaling = self.head_dim ** -0.5

        self.dropout = nn.Dropout(dropout)

    def time_delay_agg(self, values, corr):
        """
        SpeedUp version of Autocorrelation (a batch-normalization style design)
        This is for the training phase.
        """
        # corr 是指自相关系数

        # bhnl bhdl
        head = values.shape[1]
        channel = values.shape[2]
        length = values.shape[3]
        # find top k 选择k
        top_k = max(int(self.scale * math.log(length)), 1)
        # 200,42
        mean_value = torch.mean(torch.mean(corr, dim=1), dim=1)
        index = torch.topk(torch.mean(mean_value, dim=0), top_k, dim=-1)[1]
        weights = torch.stack([mean_value[:, index[i]] for i in range(top_k)], dim=-1)
        # update corr 
        tmp_corr = torch.softmax(weights, dim=-1)
        # aggregation
        tmp_values = values

        delays_agg = torch.zeros_like(values).float()

        for i in range(top_k):
            pattern = torch.roll(tmp_values, -int(index[i]), -1)
            delays_agg = delays_agg + pattern * \
                         (tmp_corr[:, i].unsqueeze(1).unsqueeze(1).unsqueeze(1).repeat(1, head, channel, length))
        return delays_agg

    def time_delay_agg_full(self, values, corr):
        """
        Standard version of Autocorrelation
        """
        batch = values.shape[0]
        head = values.shape[1]
        channel = values.shape[2]
        length = values.shape[3]
        # index init
        init_index = torch.arange(length).unsqueeze(0).unsqueeze(0).unsqueeze(0)\
            .repeat(batch, head, channel, 1).to(values.device)
        # find top k
        top_k = max(int(self.scale * math.log(length)), 1)
        weights, delay = torch.topk(corr, top_k, dim=-1)
        # update corr
        tmp_corr = torch.softmax(weights, dim=-1)
        # aggregation
        tmp_values = values.repeat(1, 1, 1, 2)
        delays_agg = torch.zeros_like(values).float()
        for i in range(top_k):
            tmp_delay = init_index + delay[..., i].unsqueeze(-1)
            pattern = torch.gather(tmp_values, dim=-1, index=tmp_delay)
            delays_agg = delays_agg + pattern * (tmp_corr[..., i].unsqueeze(-1))
        return delays_agg


    def forward(self, queries, keys, values):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.num_heads
        # 过mlp 保持维度一致 blhd
        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)
        # queries *= self.scaling
        
        # blhd-->bhdl-->bhd(l+1//2)
        q_fft = torch.fft.rfft(queries.permute(0, 2, 3, 1).contiguous(), dim=-1)
        k_fft = torch.fft.rfft(keys.permute(0, 2, 3, 1).contiguous(), dim=-1)
        # bhd(l+1//2)*bhd(l+1//2)-->bhd(l+1//2)
        res = q_fft * torch.conj(k_fft)
        # bhd(l+1//2)-->bhdl
        corr = torch.fft.irfft(res, n=L, dim=-1)


        # time delay agg
        out = self.time_delay_agg_full(values.permute(0, 2, 3, 1).contiguous(), corr).permute(0, 3, 1, 2).contiguous()
        
        # 最后
        out = out.view(B, L, -1)

        return self.out_projection(out)
