import numpy as np
import sys
import math

import oneflow as flow
import oneflow.nn as nn

sys.path.append("../")
from transformer import TransformerEncoder, TransformerEncoderLayer


class Embeddings(nn.Module):
    def __init__(self, vocab, d_model):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = flow.zeros((max_len, d_model))
        position = flow.arange(0, max_len, dtype=flow.float).unsqueeze(1)
        div_term = flow.exp(
            flow.arange(0, d_model, 2).to(flow.float) * (-math.log(10000.0) / d_model)
        ).unsqueeze(0)
        pe[:, 0::2] = flow.sin(position * div_term)
        pe[:, 1::2] = flow.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.pe = pe
        # self.register_buffer('pe', pe)

    def forward(self, x):
        self.pe = self.pe.to(x.device)
        x = x + self.pe[: x.size(0), :]
        return self.dropout(x)


class TransformerEncoderModel(nn.Module):
    def __init__(
        self,
        emb_sz,
        n_classes,
        d_model,
        nhead,
        num_encoder_layers,
        dim_feedforward,
        dropout,
        batch_first,
    ):
        super(TransformerEncoderModel, self).__init__()
        layer = TransformerEncoderLayer(
            d_model, nhead, dim_feedforward, dropout, batch_first=batch_first
        )
        self.transformer_encoder = TransformerEncoder(layer, num_encoder_layers)
        self.src_embedding = Embeddings(emb_sz, d_model)
        self.pos = PositionalEncoding(d_model, dropout)
        self.linear = nn.Linear(d_model, n_classes)

    def generate_subsequent_mask(self, tgt_len, src_len):
        mask = flow.triu(flow.ones((tgt_len, src_len)), 1).to(flow.int32)
        return mask

    def make_len_mask(self, inp):
        inp = (inp.numpy() == 0).astype(np.int32)
        inp = flow.tensor(inp, dtype=flow.int32)
        return inp

    def forward(self, src):

        src_key_padding_mask = self.make_len_mask(src).to(src.device)
        src_mask = None

        src = self.src_embedding(src)
        src = self.pos(src)
        out = self.transformer_encoder(src, src_mask, src_key_padding_mask)
        out = flow.max(out, dim=1)
        out = self.linear(out)
        return out
