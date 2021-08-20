import unittest

import numpy as np
import oneflow
import oneflow.compatible.single_client as flow
from oneflow import nn


def _FullyConnected(
    input_blob,
    input_size,
    units,
    name=None,
    weight_initializer=flow.truncated_normal(0.02),
):
    weight_blob = flow.get_variable(
        name=name + '-weight',
        shape=[input_size, units],
        dtype=input_blob.dtype,
        initializer=weight_initializer,
    )
    bias_blob = flow.get_variable(
        name=name + '-bias',
        shape=[units],
        dtype=input_blob.dtype,
        initializer=flow.constant_initializer(0.0),
    )
    output_blob = flow.matmul(input_blob, weight_blob)
    output_blob = flow.nn.bias_add(output_blob, bias_blob)
    return output_blob, weight_blob, bias_blob


def compare_with_lazy_fc():

    # lazy fully connected layer
    func_config = flow.FunctionConfig()
    func_config.default_data_type(flow.float)

    @flow.global_function("predict", func_config)
    def fc_job(input_blob: flow.typing.Numpy.Placeholder((1, 10))):
        return _FullyConnected(input_blob, input_size=10, units=4, name="dense")

    input_blobs = np.random.normal(size=(1, 10))
    lazy_res, lazy_weight, lazy_bias = fc_job(input_blobs).get()

    # eager fully connected layer
    eager_fc = nn.Linear(10, 4)
    eager_fc.weight = nn.Parameter(oneflow.tensor(lazy_weight.numpy().transpose()))
    eager_fc.bias = nn.Parameter(oneflow.tensor(lazy_bias.numpy()))

    eager_res = eager_fc(oneflow.tensor(input_blobs, dtype=oneflow.float32))

    assert np.allclose(lazy_res.numpy(), eager_res.numpy(), rtol=1e-4, atol=1e-4)


class TestFullyConnected(unittest.TestCase):
    def test_fc(self):
        compare_with_lazy_fc()


if __name__ == "__main__":

    unittest.main()
