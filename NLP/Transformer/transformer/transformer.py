r"""
reference to PyTorch

Transformer                 in torch.nn.modules.transformer
TransformerEncoder          in torch.nn.modules.transformer
TransformerDecoder          in torch.nn.modules.transformer
TransformerEncoderLayer     in torch.nn.modules.transformer
TransformerDecoderLayer     in torch.nn.modules.transformer
"""
import copy
from typing import Optional, Any

import oneflow as flow
from oneflow import Tensor
from oneflow.nn import Module, Dropout, Linear, LayerNorm, ModuleList
from oneflow.nn.init import xavier_uniform_

from .multihead_attention import MultiheadAttention


class Transformer(Module):
    def __init__(
        self,
        d_model: int = 512,
        nhead: int = 8,
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 6,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: str = "relu",
        custom_encoder: Optional[Any] = None,
        custom_decoder: Optional[Any] = None,
        layer_norm_eps: float = 1e-5,
        batch_first: bool = False,
        norm_first: bool = False,
    ) -> None:
        super(Transformer, self).__init__()

        if custom_encoder is not None:
            self.encoder = custom_encoder
        else:
            encoder_norm = LayerNorm(d_model, eps=layer_norm_eps)
            encoder_layer = TransformerEncoderLayer(
                d_model,
                nhead,
                dim_feedforward,
                dropout,
                activation,
                layer_norm_eps,
                batch_first,
                norm_first,
            )
            self.encoder = TransformerEncoder(
                encoder_layer, num_encoder_layers, encoder_norm
            )

        if custom_decoder is not None:
            self.decoder = custom_decoder
        else:
            decoder_norm = LayerNorm(d_model, eps=layer_norm_eps)
            decoder_layer = TransformerDecoderLayer(
                d_model,
                nhead,
                dim_feedforward,
                dropout,
                activation,
                layer_norm_eps,
                batch_first,
                norm_first,
            )
            self.decoder = TransformerDecoder(
                decoder_layer, num_decoder_layers, decoder_norm
            )

        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead

        self.batch_first = batch_first

    def forward(
        self,
        src: Tensor,
        tgt: Tensor,
        src_mask: Optional[Tensor] = None,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
    ) -> Tensor:

        if not self.batch_first and src.size(1) != tgt.size(1):
            raise RuntimeError("the batch number of src and tgt must be equal")
        elif self.batch_first and src.size(0) != tgt.size(0):
            raise RuntimeError("the batch number of src and tgt must be equal")

        if src.size(2) != self.d_model or tgt.size(2) != self.d_model:
            raise RuntimeError(
                "the feature number of src and tgt must be equal to d_model"
            )

        memory = self.encoder(src, src_mask, src_key_padding_mask)
        output = self.decoder(
            tgt,
            memory,
            tgt_mask,
            memory_mask,
            tgt_key_padding_mask,
            memory_key_padding_mask,
        )
        return output

    def generate_square_subsequent_mask(self, sz: int) -> Tensor:
        return flow.triu(flow.ones((sz, sz)), diagonal=1).to(flow.int32)

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                xavier_uniform_(p)


class TransformerEncoder(Module):
    __constants__ = ["norm"]

    def __init__(self, encoder_layer, num_layers, norm=None):
        super(TransformerEncoder, self).__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(
        self,
        src: Tensor,
        mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
    ) -> Tensor:

        output = src
        for mod in self.layers:
            output = mod(output, mask, src_key_padding_mask)

        if self.norm is not None:
            output = self.norm(output)

        return output


class TransformerDecoder(Module):
    __constants__ = ["norm"]

    def __init__(self, decoder_layer, num_layers, norm=None):
        super(TransformerDecoder, self).__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(
        self,
        tgt: Tensor,
        memory: Tensor,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
    ) -> Tensor:

        output = tgt
        for mod in self.layers:
            output = mod(
                output,
                memory,
                tgt_mask,
                memory_mask,
                tgt_key_padding_mask,
                memory_key_padding_mask,
            )

        if self.norm is not None:
            output = self.norm(output)

        return output


class TransformerEncoderLayer(Module):
    __constants__ = ["batch_first", "norm_first"]

    def __init__(
        self,
        d_model,
        nhead,
        dim_feedforward=2048,
        dropout=0.1,
        activation="relu",
        layer_norm_eps=1e-5,
        batch_first=False,
        norm_first=False,
    ) -> None:
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=batch_first
        )
        # Implementation of Feedforward model
        self.linear1 = Linear(d_model, dim_feedforward)
        self.dropout = Dropout(dropout)
        self.linear2 = Linear(dim_feedforward, d_model)

        self.norm_first = norm_first
        self.norm1 = LayerNorm(d_model, eps=layer_norm_eps)
        self.norm2 = LayerNorm(d_model, eps=layer_norm_eps)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def forward(
        self,
        src: Tensor,
        src_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
    ) -> Tensor:

        if self.norm_first:
            src = self.norm1(src)
            src2 = self.self_attn(src, src, src, src_key_padding_mask, True, src_mask)[
                0
            ]
            src = src + self.dropout1(src2)
            src = self.norm2(src)
            src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
            src = src + self.dropout2(src2)
            return src

        # norm last
        src2 = self.self_attn(src, src, src, src_key_padding_mask, True, src_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src


class TransformerDecoderLayer(Module):
    __constants__ = ["batch_first", "norm_first"]

    def __init__(
        self,
        d_model,
        nhead,
        dim_feedforward=2048,
        dropout=0.1,
        activation="relu",
        layer_norm_eps=1e-5,
        batch_first=False,
        norm_first=False,
    ) -> None:
        super(TransformerDecoderLayer, self).__init__()
        self.self_attn = MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=batch_first
        )
        self.multihead_attn = MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=batch_first
        )
        # Implementation of Feedforward model
        self.linear1 = Linear(d_model, dim_feedforward)
        self.dropout = Dropout(dropout)
        self.linear2 = Linear(dim_feedforward, d_model)

        self.norm_first = norm_first
        self.norm1 = LayerNorm(d_model, eps=layer_norm_eps)
        self.norm2 = LayerNorm(d_model, eps=layer_norm_eps)
        self.norm3 = LayerNorm(d_model, eps=layer_norm_eps)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)
        self.dropout3 = Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def forward(
        self,
        tgt: Tensor,
        memory: Tensor,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        if self.norm_first:
            tgt = self.norm1(tgt)
            tgt2 = self.self_attn(tgt, tgt, tgt, tgt_key_padding_mask, True, tgt_mask)[
                0
            ]
            tgt = tgt + self.dropout1(tgt2)
            tgt = self.norm2(tgt)
            tgt2 = self.multihead_attn(
                tgt, memory, memory, memory_key_padding_mask, True, memory_mask
            )[0]
            tgt = tgt + self.dropout2(tgt2)
            tgt = self.norm3(tgt)
            tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
            tgt = tgt + self.dropout3(tgt2)
            return tgt

        # norm last
        tgt2 = self.self_attn(tgt, tgt, tgt, tgt_key_padding_mask, True, tgt_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt2 = self.multihead_attn(
            tgt, memory, memory, memory_key_padding_mask, True, memory_mask
        )[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt


def _get_clones(module, N):
    return ModuleList([copy.deepcopy(module) for _ in range(N)])


def _get_activation_fn(activation):
    if activation == "relu":
        return flow.nn.functional.relu
    elif activation == "gelu":
        return flow.nn.functional.gelu

    raise RuntimeError("activation should be relu/gelu, not {}".format(activation))
