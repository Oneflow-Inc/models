import oneflow as flow
import oneflow.nn as nn


def get_length_mask(tensor, tensor_length):
    b, t, _ = tensor.size()
    mask = tensor.new_zeros([b, t], dtype=flow.int8)
    for i, length in enumerate(tensor_length):
        length = length.item()
        mask[i].narrow(0, 0, length).fill_(1)
    return mask > 0
