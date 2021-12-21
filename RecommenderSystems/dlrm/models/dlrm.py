from collections import OrderedDict
import oneflow as flow
import oneflow.nn as nn
from typing import Any


__all__ = ["make_dlrm_module"]


class Dense(nn.Module):
    def __init__(
        self, in_features: int, out_features: int, dropout_rate: float = 0.5
    ) -> None:
        super(Dense, self).__init__()
        self.features = nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
        )
        for name, param in self.named_parameters():
            if name.endswith("weight"):
                nn.init.xavier_uniform_(param)
            elif name.endswith("bias"):
                nn.init.zeros_(param)

    def forward(self, x: flow.Tensor) -> flow.Tensor:
        x = self.features(x)
        return x


class Embedding(nn.Embedding):
    def __init__(self, vocab_size, embed_size):
        super(Embedding, self).__init__(vocab_size, embed_size, padding_idx=0)
        for param in self.parameters():
            nn.init.uniform_(param, a=-0.05, b=0.05)


class ConsistentDLRM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embedding_vec_size: int = 16,
        num_sparse_fields: int = 26,
        num_dense_fields: int = 13,
        hidden_size: int = 1024,
        hidden_units_num: int = 7,
        dropout_rate: float = 0.5,
    ):
        super(ConsistentDLRM, self).__init__()

        self.embedding = Embedding(vocab_size, embedding_vec_size // flow.env.get_world_size())
        self.embedding.to_consistent(flow.env.all_device_placement("cuda"), flow.sbp.split(1))
        feature_size = (
            embedding_vec_size * num_sparse_fields
            + num_dense_fields
        )
        self.linear_layers = nn.Sequential(
            OrderedDict(
                [
                    (
                        f"fc{i}",
                        Dense(
                            feature_size if i == 0 else hidden_size,
                            hidden_size,
                            dropout_rate,
                        ),
                    )
                    for i in range(hidden_units_num)
                ]
            )
        )
        self.linear_layers.to_consistent(flow.env.all_device_placement("cuda"), flow.sbp.broadcast)
        self.scores = nn.Linear(hidden_size, 1)
        self.scores.to_consistent(flow.env.all_device_placement("cuda"), flow.sbp.broadcast)
        self.sigmoid = nn.Sigmoid()
        self.sigmoid.to_consistent(flow.env.all_device_placement("cuda"), flow.sbp.broadcast)

    def forward(self, dense_fields, sparse_fields) -> flow.Tensor:
        sparse_fields = sparse_fields.to_consistent(sbp=flow.sbp.broadcast)
        embedding = self.embedding(sparse_fields)
        embedding = embedding.to_consistent(sbp=flow.sbp.split(0), grad_sbp=flow.sbp.split(2))
        embedding = embedding.view(-1, embedding.shape[-1] * embedding.shape[-2])
        features = flow.cat([embedding, dense_fields], dim=1)
        features = self.linear_layers(features)
        scores = self.scores(features)
        return self.sigmoid(scores)


class LocalDLRM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embedding_vec_size: int = 16,
        num_sparse_fields: int = 26,
        num_dense_fields: int = 13,
        hidden_size: int = 1024,
        hidden_units_num: int = 7,
        dropout_rate: float = 0.5,
    ):
        super(LocalDLRM, self).__init__()
        self.embedding = Embedding(vocab_size, embedding_vec_size)
        feature_size = (
            embedding_vec_size * num_sparse_fields + num_dense_fields
        )
        self.linear_layers = nn.Sequential(
            OrderedDict(
                [
                    (
                        f"fc{i}",
                        Dense(
                            feature_size if i == 0 else hidden_size,
                            hidden_size,
                            dropout_rate,
                        ),
                    )
                    for i in range(hidden_units_num)
                ]
            )
        )
        self.scores = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()


    def forward(self, dense_fields, sparse_fields) -> flow.Tensor:
        embedding = self.embedding(sparse_fields)
        embedding = embedding.view(-1, embedding.shape[-1] * embedding.shape[-2])
        features = flow.cat([embedding, dense_fields], dim=1)
        features = self.linear_layers(features)
        scores = self.scores(features)
        return self.sigmoid(scores)


def make_dlrm_module(args, is_consistent):
    if is_consistent:
        model = ConsistentDLRM(
            vocab_size=args.vocab_size,
            embedding_vec_size=args.embedding_vec_size,
            num_sparse_fields=args.num_sparse_fields,
            num_dense_fields=args.num_dense_fields,
            hidden_size=args.hidden_size,
            hidden_units_num=args.hidden_units_num,
            dropout_rate=args.dropout_rate,
        )
    else:
        model = LocalDLRM(
            vocab_size=args.vocab_size,
            embedding_vec_size=args.embedding_vec_size,
            num_sparse_fields=args.num_sparse_fields,
            num_dense_fields=args.num_dense_fields,
            hidden_size=args.hidden_size,
            hidden_units_num=args.hidden_units_num,
            dropout_rate=args.dropout_rate,
        )
        model = model.to("cuda")
    return model
