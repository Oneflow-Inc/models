import numpy as np
import oneflow as flow
import oneflow.nn as nn
from typing import Optional

from .dev_ops import LayerNorm


def shift_tokens_right(
    input_ids: flow.Tensor, pad_token_id: int, decoder_start_token_id: int
):
    """
    Shift input ids one token to the right.
    """
    shifted_input_ids = flow.zeros_like(input_ids)
    shifted_input_ids[:, 1:] = input_ids[:, :-1].clone()

    # shifted_input_ids[:, 0] = decoder_start_token_id
    # tensor assignment in oneflow:
    shifted_input_ids[:, 0] = flow.tensor(
        decoder_start_token_id,
        dtype=shifted_input_ids.dtype,
        device=shifted_input_ids.device,
    )

    assert pad_token_id is not None, "self.model.pad_token_id has to be defined."
    # replace possible -100 values in labels by `pad_token_id`
    # masked
    shifted_input_ids = (
        shifted_input_ids.to(flow.float)
        .masked_fill(shifted_input_ids.eq(-100).to(flow.int32), pad_token_id)
        .to(flow.int32)
    )

    return shifted_input_ids


def _make_causal_mask(
    input_ids_shape: flow.Size,
    dtype: flow.dtype,
    device: flow.device,
    past_key_values_length: int = 0,
):
    """
    Make causal mask used for bi-directional self-attention.
    """
    bsz, tgt_len = input_ids_shape
    mask = flow.ones((tgt_len, tgt_len)) * float("-inf")
    mask_cond = flow.arange(mask.size(-1))
    mask = mask.masked_fill(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
    mask = mask.to(dtype)

    if past_key_values_length > 0:
        mask = flow.cat(
            [flow.zeros(tgt_len, past_key_values_length, dtype=dtype), mask], dim=-1
        )
    return (
        mask[None, None, :, :]
        .expand(bsz, 1, tgt_len, tgt_len + past_key_values_length)
        .to(device)
    )


def _expand_mask(mask: flow.Tensor, dtype: flow.dtype, tgt_len: Optional[int] = None):
    """
    Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.
    """
    bsz, src_len = mask.size()
    tgt_len = tgt_len if tgt_len is not None else src_len

    expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)

    inverted_mask = 1.0 - expanded_mask

    return inverted_mask.masked_fill(inverted_mask.to(flow.int32), -1e9)


def init_weights(module):

    if isinstance(module, nn.Linear):
        module.weight.data.normal_(mean=0.0, std=0.02)
        if module.bias is not None:
            module.bias.data.fill_(0.0)
    elif isinstance(module, nn.Embedding):
        module.weight.data.normal_(mean=0.0, std=0.02)
        if module.padding_idx is not None:
            module.weight.data[module.padding_idx].fill_(0.01)
    elif isinstance(module, nn.LayerNorm):
        module.bias.data.fill_(0.0)
        module.weight.data.fill_(1.0)
    elif isinstance(module, LayerNorm):
        module.bias.data.fill_(0.0)
        module.weight.data.fill_(1.0)


# for tensor.unique


def tensor_unique(tensor):

    _np = tensor.numpy()
    return flow.tensor(np.unique(_np), dtype=tensor.dtype, device=tensor.device)
