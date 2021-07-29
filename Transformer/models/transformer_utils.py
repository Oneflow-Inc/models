# reference to py
import oneflow as flow

from oneflow.F import dropout
from oneflow import Tensor
from typing import Optional, Tuple, List
import math

# Some functions in torch.nn.functional


def pad(input, padding, mode='constant'):

    origin_dim = input.dim()
    for i in range(4 - origin_dim):
        input = input.unsqueeze(0)

    input = flow.F.pad(input, padding, mode=mode)

    for i in range(4 - origin_dim):
        input = input.squeeze(0)

    return input


def linear(input, weight, bias=None):
    r"""
    Shape:
        input: (N, *, in_features)
        weight: (out_features, in_features)
        bias: (N, *, out_features)
        Output: (N, *, out_features)
    """

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
    r"""
    Computes scaled dot product attention on query, key and value tensors, using
    an optional attention mask if passed, and applying dropout if a probability
    greater than 0.0 is specified.
    Returns a tensor pair containing attended values and attention weights.
    Args:
        q, k, v: query, key and value tensors. See Shape section for shape details.
        attn_mask: optional tensor containing mask values to be added to calculated
            attention. May be 2D or 3D; see Shape section for details.
        dropout_p: dropout probability. If greater than 0.0, dropout is applied.
    Shape:
        - q: :math:`(B, Nt, E)` where B is batch size, Nt is the target sequence length,
            and E is embedding dimension.
        - key: :math:`(B, Ns, E)` where B is batch size, Ns is the source sequence length,
            and E is embedding dimension.
        - value: :math:`(B, Ns, E)` where B is batch size, Ns is the source sequence length,
            and E is embedding dimension.
        - attn_mask: either a 3D tensor of shape :math:`(B, Nt, Ns)` or a 2D tensor of
            shape :math:`(Nt, Ns)`.
        - Output: attention values have shape :math:`(B, Nt, E)`; attention weights
            have shape :math:`(B, Nt, Ns)`
    """
    B, Nt, E = q.shape
    q = q / math.sqrt(E)
    # (B, Nt, E) x (B, E, Ns) -> (B, Nt, Ns)
    attn = flow.bmm(q, k.transpose(-2, -1))
    if attn_mask is not None:
        attn += attn_mask
    attn = flow.softmax(attn, dim=-1)
    if dropout_p > 0.0:
        attn = dropout(attn, p=dropout_p)
    # (B, Nt, Ns) x (B, Ns, E) -> (B, Nt, E)
    output = flow.bmm(attn, v)
    return output, attn


def _in_projection_packed(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    w: Tensor,
    b: Optional[Tensor] = None,
) -> List[Tensor]:
    r"""
    Shape:
        Inputs:
        - q: :math:`(..., E)` where E is the embedding dimension
        - k: :math:`(..., E)` where E is the embedding dimension
        - v: :math:`(..., E)` where E is the embedding dimension
        - w: :math:`(E * 3, E)` where E is the embedding dimension
        - b: :math:`E * 3` where E is the embedding dimension
        Output:
        - in output list :math:`[q', k', v']`, each output tensor will have the
            same shape as the corresponding input tensor.
    """
    E = q.size(-1)
    if k is v:
        if q is k:
            # self-attention
            # dim=-1的时候chunk不起作用
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
                #b_q, b_kv = b.split([E, E * 2])
                b_q, b_k, b_v = b.chunk(3, dim=0)
                b_kv = flow.cat([b_k, b_v])
            res = linear(k, w_kv, b_kv)
            chunk_dim = len(res.shape)
            # 似乎与return linear(q, w_q, b_q), linear(k, w_k, b_k), linear(k, w_v, b_v)等效？
            return [linear(q, w_q, b_q)] + res.chunk(2, dim=chunk_dim - 1)
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
    r"""
    Performs the in-projection step of the attention operation. This is simply
    a triple of linear projections, with shape constraints on the weights which
    ensure embedding dimension uniformity in the projected outputs.
    Output is a triple containing projection tensors for query, key and value.
    Args:
        q, k, v: query, key and value tensors to be projected.
        w_q, w_k, w_v: weights for q, k and v, respectively.
        b_q, b_k, b_v: optional biases for q, k and v, respectively.
    Shape:
        Inputs:
        - q: :math:`(Qdims..., Eq)` where Eq is the query embedding dimension and Qdims are any
            number of leading dimensions.
        - k: :math:`(Kdims..., Ek)` where Ek is the key embedding dimension and Kdims are any
            number of leading dimensions.
        - v: :math:`(Vdims..., Ev)` where Ev is the value embedding dimension and Vdims are any
            number of leading dimensions.
        - w_q: :math:`(Eq, Eq)`
        - w_k: :math:`(Eq, Ek)`
        - w_v: :math:`(Eq, Ev)`
        - b_q: :math:`(Eq)`
        - b_k: :math:`(Eq)`
        - b_v: :math:`(Eq)`
        Output: in output triple :math:`(q', k', v')`,
         - q': :math:`[Qdims..., Eq]`
         - k': :math:`[Kdims..., Eq]`
         - v': :math:`[Vdims..., Eq]`
    """
    Eq, Ek, Ev = q.size(-1), k.size(-1), v.size(-1)
    assert w_q.shape == (
        Eq, Eq), f"expecting query weights shape of {(Eq, Eq)}, but got {w_q.shape}"
    assert w_k.shape == (
        Eq, Ek), f"expecting key weights shape of {(Eq, Ek)}, but got {w_k.shape}"
    assert w_v.shape == (
        Eq, Ev), f"expecting value weights shape of {(Eq, Ev)}, but got {w_v.shape}"
    assert b_q is None or b_q.shape == (
        Eq,), f"expecting query bias shape of {(Eq,)}, but got {b_q.shape}"
    assert b_k is None or b_k.shape == (
        Eq,), f"expecting key bias shape of {(Eq,)}, but got {b_k.shape}"
    assert b_v is None or b_v.shape == (
        Eq,), f"expecting value bias shape of {(Eq,)}, but got {b_v.shape}"
    return linear(q, w_q, b_q), linear(k, w_k, b_k), linear(v, w_v, b_v)
