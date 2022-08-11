r"""
reference to PyTorch

linear                          in torch.nn.functional
_scaled_dot_product_attention   in torch.nn.functional
_in_projection_packed           in torch.nn.functional
_in_projection                  in torch.nn.functional
"""
import oneflow as flow

from oneflow import Tensor
from typing import Optional, Tuple, List
import math


def linear(input, weight, bias=None):
    if input.dim() == 2 and bias is not None:
        # fused op is marginally faster
        ret = flow.addmm(bias, input, weight.transpose(0, 1))
    else:
        output = input.matmul(weight.transpose(0, 1))
        if bias is not None:
            output += bias
        ret = output
    return ret


def _scaled_dot_product_attention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    attn_mask: Optional[Tensor] = None,
    dropout_p: float = 0.0,
) -> Tuple[Tensor, Tensor]:
    B, Nt, E = q.shape
    q = q / math.sqrt(E)
    # (B, Nt, E) x (B, E, Ns) -> (B, Nt, Ns)
    attn = flow.bmm(q, k.transpose(-2, -1))
    if attn_mask is not None:
        attn += attn_mask
    attn = flow.softmax(attn, dim=-1)
    if dropout_p > 0.0:
        attn = flow.nn.functional.dropout(attn, p=dropout_p)
    # (B, Nt, Ns) x (B, Ns, E) -> (B, Nt, E)
    output = flow.bmm(attn, v)
    return output, attn


def _in_projection_packed(
    q: Tensor, k: Tensor, v: Tensor, w: Tensor, b: Optional[Tensor] = None,
) -> List[Tensor]:
    E = q.size(-1)
    if k is v:
        if q is k:
            # self-attention
            # Chunk does not work when dim=-1
            res = linear(q, w, b)
            chunk_dim = len(res.shape)
            return res.chunk(3, dim=chunk_dim - 1)
        else:
            # encoder-decoder attention
            # w_q, w_kv = w.split([E, E * 2])
            w_q, w_k, w_v = w.chunk(3, dim=0)
            w_kv = flow.cat([w_k, w_v])
            if b is None:
                b_q = b_kv = None
            else:
                # b_q, b_kv = b.split([E, E * 2])
                b_q, b_k, b_v = b.chunk(3, dim=0)
                b_kv = flow.cat([b_k, b_v])
            res = linear(k, w_kv, b_kv)
            chunk_dim = len(res.shape)
            return (linear(q, w_q, b_q),) + res.chunk(2, dim=chunk_dim - 1)
    else:
        w_q, w_k, w_v = w.chunk(3, dim=0)
        if b is None:
            b_q = b_k = b_v = None
        else:
            b_q, b_k, b_v = b.chunk(3, dim=0)
        return linear(q, w_q, b_q), linear(k, w_k, b_k), linear(v, w_v, b_v)


def _in_projection(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    w_q: Tensor,
    w_k: Tensor,
    w_v: Tensor,
    b_q: Optional[Tensor] = None,
    b_k: Optional[Tensor] = None,
    b_v: Optional[Tensor] = None,
) -> Tuple[Tensor, Tensor, Tensor]:
    Eq, Ek, Ev = q.size(-1), k.size(-1), v.size(-1)
    assert w_q.shape == (
        Eq,
        Eq,
    ), f"expecting query weights shape of {(Eq, Eq)}, but got {w_q.shape}"
    assert w_k.shape == (
        Eq,
        Ek,
    ), f"expecting key weights shape of {(Eq, Ek)}, but got {w_k.shape}"
    assert w_v.shape == (
        Eq,
        Ev,
    ), f"expecting value weights shape of {(Eq, Ev)}, but got {w_v.shape}"
    assert b_q is None or b_q.shape == (
        Eq,
    ), f"expecting query bias shape of {(Eq,)}, but got {b_q.shape}"
    assert b_k is None or b_k.shape == (
        Eq,
    ), f"expecting key bias shape of {(Eq,)}, but got {b_k.shape}"
    assert b_v is None or b_v.shape == (
        Eq,
    ), f"expecting value bias shape of {(Eq,)}, but got {b_v.shape}"
    return linear(q, w_q, b_q), linear(k, w_k, b_k), linear(v, w_v, b_v)
