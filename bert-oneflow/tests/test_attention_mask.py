import unittest

import numpy as np
import oneflow
import oneflow.compatible.single_client as flow


def _CreateAttentionMaskFromInputMask(
    to_mask_blob, from_seq_length=10, to_seq_length=10
):
    output = flow.cast(to_mask_blob, dtype=flow.float)
    output = flow.reshape(output, [-1, 1, to_seq_length])
    zeros = flow.constant(0.0, dtype=flow.float, shape=[from_seq_length, to_seq_length])
    output = zeros + output
    return output


def _CreateAddrFromAttentionMask(
    attention_mask_blob, from_seq_length=10, to_seq_length=10
):
    attention_mask_blob = flow.reshape(
        attention_mask_blob, [-1, 1, from_seq_length, to_seq_length]
    )
    attention_mask_blob = flow.cast(attention_mask_blob, dtype=flow.float)
    addr_blob = (attention_mask_blob - 1.0) * 10000.0
    return addr_blob


def create_attention_mask(to_mask_blob, from_seq_length=10, to_seq_length=10):
    output = oneflow.cast(to_mask_blob, dtype=oneflow.float32)
    output = oneflow.reshape(output, [-1, 1, to_seq_length])
    # broadcast `from_tensor` from 2D to 3D
    zeros = oneflow.zeros((from_seq_length, to_seq_length), dtype=oneflow.float32)
    output = output + zeros

    attention_mask = oneflow.reshape(output, [-1, 1, from_seq_length, to_seq_length])
    attention_mask = oneflow.cast(attention_mask, dtype=oneflow.float32)
    addr_blob = (attention_mask - 1.0) * 10000.0
    return addr_blob


def compare_with_lazy_attention_mask():

    # lazy attention mask
    func_config = flow.FunctionConfig()
    func_config.default_data_type(flow.float)

    @flow.global_function("predict", func_config)
    def attention_mask_job(
        input_blob: flow.typing.Numpy.Placeholder((1, 10,), dtype=flow.int32)
    ):
        attention_mask_blob = _CreateAttentionMaskFromInputMask(input_blob)
        addr_blob = _CreateAddrFromAttentionMask(attention_mask_blob)
        return addr_blob

    input_mask = np.array([[1, 1, 1, 1, 1, 1, 0, 0, 0, 0]], dtype=np.int32)
    lazy_res = attention_mask_job(input_mask).get()

    # eager attention mask
    eager_res = create_attention_mask(oneflow.tensor(input_mask))

    assert np.allclose(lazy_res.numpy(), eager_res.numpy(), rtol=1e-4, atol=1e-4)


class TestAttentionMask(unittest.TestCase):
    def test_embedding(self):
        compare_with_lazy_attention_mask()


if __name__ == "__main__":
    unittest.main()
