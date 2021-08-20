import unittest

import numpy as np
import oneflow
import oneflow.compatible.single_client as flow
from oneflow import nn


def _EmbeddingLookup(
    input_ids_blob,
    vocab_size,
    embedding_size=128,
    initializer_range=0.02,
    word_embedding_name="word_embeddings",
):
    embedding_table = flow.get_variable(
        name=word_embedding_name,
        shape=[vocab_size, embedding_size],
        dtype=flow.float,
        initializer=flow.truncated_normal(initializer_range),
    )
    output = flow.gather(params=embedding_table, indices=input_ids_blob, axis=0)
    return output, embedding_table


def compare_with_lazy_embedding():

    # lazy embedding table
    func_config = flow.FunctionConfig()
    func_config.default_data_type(flow.float)

    @flow.global_function("predict", func_config)
    def embedding_job(
        input_blob: flow.typing.Numpy.Placeholder((1, 10,), dtype=flow.int32)
    ):
        return _EmbeddingLookup(input_blob, vocab_size=1000)

    input_blobs = np.array([[0, 1, 1, 3, 2, 4, 7, 10, 11, 8]], dtype=np.int32)
    lazy_res, lazy_embedding = embedding_job(input_blobs).get()
    # print(f"result is {lazy_res.numpy()}, embedding is {lazy_embedding.numpy()}")

    # eager embedding table
    eager_embedding_table = nn.Embedding(1000, 128)
    eager_embedding_table.weight = nn.Parameter(oneflow.tensor(lazy_embedding.numpy()))
    eager_res = eager_embedding_table(oneflow.tensor(input_blobs))

    assert np.allclose(lazy_res.numpy(), eager_res.numpy(), rtol=1e-4, atol=1e-4)


class TestEmbeddingTable(unittest.TestCase):
    def test_embedding(self):
        compare_with_lazy_embedding()


if __name__ == "__main__":
    unittest.main()
