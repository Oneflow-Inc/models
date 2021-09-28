from transformer import Transformer
import numpy as np
import math
import sys

import oneflow as flow
import oneflow.nn as nn

sys.path.append("../")

TO_CUDA = True


def to_cuda(tensor, flag=TO_CUDA, where="cuda"):
    if flag:
        return tensor.to(where)
    else:
        return tensor


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
        self.pe = flow.nn.Parameter(pe, requires_grad=False)

    def forward(self, x):
        x = x + self.pe[: x.size(0), :]
        return self.dropout(x)


class TransformerModel(nn.Module):
    def __init__(
        self,
        input_sz,
        output_sz,
        d_model,
        nhead,
        num_encoder_layers,
        num_decoder_layers,
        dim_feedforward,
        dropout,
    ):
        super(TransformerModel, self).__init__()
        self.transformer = Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=False,
        )
        self.softmax = nn.Softmax(dim=2)
        self.linear = nn.Linear(d_model, output_sz)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        self.pos_decoder = PositionalEncoding(d_model, dropout)
        self.src_embedding = Embeddings(input_sz, d_model)
        self.tgt_embedding = Embeddings(output_sz, d_model)

    def generate_subsequent_mask(self, tgt_len, src_len):
        mask = flow.triu(flow.ones((tgt_len, src_len)), 1).to(flow.int32)
        return mask

    def make_len_mask(self, inp):
        inp_mask = (inp.numpy() == 0).astype(np.int32)
        inp_mask = flow.tensor(inp_mask, dtype=flow.int32)
        return inp_mask.transpose(0, 1)

    def forward(
        self,
        src,
        tgt,
        src_mask=None,
        tgt_mask=None,
        memory_mask=None,
        src_key_padding_mask=None,
        tgt_key_padding_mask=None,
        memory_key_padding_mask=None,
    ):
        if tgt_mask is None:
            tgt_mask = self.generate_subsequent_mask(tgt.shape[0], tgt.shape[0])
            tgt_mask = to_cuda(tgt_mask, where=tgt.device)

        src_key_padding_mask = self.make_len_mask(src)
        src_key_padding_mask = to_cuda(src_key_padding_mask, where=tgt.device)

        tgt_key_padding_mask = None

        src = self.src_embedding(src)
        src = self.pos_encoder(src)

        tgt = self.tgt_embedding(tgt)
        tgt = self.pos_decoder(tgt)

        out = self.transformer(
            src,
            tgt,
            src_mask,
            tgt_mask,
            memory_mask,
            src_key_padding_mask,
            tgt_key_padding_mask,
            memory_key_padding_mask,
        )
        out = self.linear(out)
        return out
