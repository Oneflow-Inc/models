import oneflow.nn as nn
import oneflow as flow
import math
import numpy as np


class Attention(nn.Module):
    """
    Compute 'Scaled Dot Product Attention
    """

    def __init__(self):
        super().__init__()
        self.softmax = flow.nn.Softmax(dim=-1)

    def forward(self, query, key, value, mask=None, dropout=None):
        x = flow.matmul(query, key.transpose(-2, -1))
        scores = x / math.sqrt(query.size(-1))

        if mask is not None:
            # mask = flow.Tensor(
            #     (mask.numpy() == 0).astype(np.int8),
            #     dtype=flow.int,
            #     device=scores.device,
            # )

            # ==0 换成equal 0. 要么加scalar版本，要么等chengcheng的PR
            
            # mask = flow.Tensor(
            #     (mask.numpy() == 0).astype(np.int8),
            #     dtype=flow.int,
            #     device=scores.device,
            # )

            # scores = scores.masked_fill(mask, -1e9)
            print("==")
            
        p_attn = self.softmax(scores)

        if dropout is not None:
            p_attn = dropout(p_attn)

        return flow.matmul(p_attn, value), p_attn
