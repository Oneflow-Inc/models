import unittest

import numpy as np
import oneflow
import oneflow.compatible.single_client as flow
from oneflow import nn


def CreateInitializer(std):
    return flow.truncated_normal(std)


def _EmbeddingPostprocessor(
    input_blob,
    seq_length,
    embedding_size,
    use_token_type=True,
    token_type_ids_blob=None,
    token_type_vocab_size=2,
    token_type_embedding_name="token_type_embeddings",
    use_position_embeddings=True,
    position_embedding_name="position_embeddings",
    initializer_range=0.02,
    max_position_embeddings=512,
):
    output = input_blob

    if use_token_type:
        assert token_type_ids_blob is not None
        token_type_table = flow.get_variable(
            name=token_type_embedding_name,
            shape=[token_type_vocab_size, embedding_size],
            dtype=input_blob.dtype,
            initializer=CreateInitializer(initializer_range),
        )
        token_type_embeddings = flow.gather(
            params=token_type_table, indices=token_type_ids_blob, axis=0
        )
        output = output + token_type_embeddings

    if use_position_embeddings:
        position_table = flow.get_variable(
            name=position_embedding_name,
            shape=[1, max_position_embeddings, embedding_size],
            dtype=input_blob.dtype,
            initializer=CreateInitializer(initializer_range),
        )
        assert seq_length <= max_position_embeddings
        if seq_length != max_position_embeddings:
            position_table_res = flow.slice(
                position_table, begin=[None, 0, 0], size=[None, seq_length, -1]
            )
        output = output + position_table_res

    return output, token_type_table, position_table


class EmbeddingPostProcessor(nn.Module):
    def __init__(
        self,
        seq_length,
        embedding_size,
        token_type_vocab_size=2,
        max_position_embeddings=512,
    ):
        super().__init__()
        self.seq_length = seq_length
        self.token_embedding_table = nn.Embedding(token_type_vocab_size, embedding_size)

        self.position_embedding_table = nn.Parameter(
            oneflow.Tensor(1, max_position_embeddings, embedding_size)
        )

    def forward(self, input_blob, token_type_ids_blob):
        token_type_embedding = self.token_embedding_table(token_type_ids_blob)
        position_embedding = self.position_embedding_table[:, : self.seq_length]
        return input_blob + token_type_embedding + position_embedding

    def weight_init(self, token_table, position_table):
        self.token_embedding_table.weight = nn.Parameter(oneflow.tensor(token_table))
        self.position_embedding_table = nn.Parameter(oneflow.tensor(position_table))


def compare_with_lazy_embed_post():

    # lazy embedding post processor
    func_config = flow.FunctionConfig()
    func_config.default_data_type(flow.float)

    @flow.global_function("predict", func_config)
    def embedpost_job(
        input_blob: flow.typing.Numpy.Placeholder((1, 10, 128)),
        token_type_blob: flow.typing.Numpy.Placeholder((1, 10), dtype=flow.int32),
    ):
        return _EmbeddingPostprocessor(
            input_blob,
            seq_length=10,
            embedding_size=128,
            token_type_ids_blob=token_type_blob,
        )

    input_blobs = np.random.normal(size=(1, 10, 128))
    token_type_ids = np.array([[0, 0, 0, 0, 0, 0, 1, 1, 1, 1]], dtype=np.int32)
    lazy_res, token_type_table, position_table = embedpost_job(
        input_blobs, token_type_ids
    ).get()

    # eager embedding post processor
    eager_embedpost = EmbeddingPostProcessor(10, 128)
    eager_embedpost.weight_init(token_type_table.numpy(), position_table.numpy())
    eager_res = eager_embedpost(
        oneflow.tensor(input_blobs, dtype=oneflow.float32),
        oneflow.tensor(token_type_ids, dtype=oneflow.int32),
    )

    assert np.allclose(lazy_res.numpy(), eager_res.numpy(), rtol=1e-4, atol=1e-4)


class TestEmbeddingPost(unittest.TestCase):
    def test_fc(self):
        compare_with_lazy_embed_post()


if __name__ == "__main__":
    unittest.main()
