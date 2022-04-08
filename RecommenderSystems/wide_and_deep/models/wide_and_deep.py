from collections import OrderedDict
import oneflow as flow
import oneflow.nn as nn
from typing import Any


__all__ = ["make_wide_and_deep_module"]


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


class GlobalWideAndDeep(nn.Module):
    def __init__(
        self,
        wide_vocab_size: int,
        deep_vocab_size: int,
        deep_embedding_vec_size: int = 16,
        num_deep_sparse_fields: int = 26,
        num_dense_fields: int = 13,
        hidden_size: int = 1024,
        hidden_units_num: int = 7,
        deep_dropout_rate: float = 0.5,
    ):
        super(GlobalWideAndDeep, self).__init__()

        self.wide_embedding = Embedding(wide_vocab_size // flow.env.get_world_size(), 1)
        self.wide_embedding.to_global(
            flow.env.all_device_placement("cuda"), flow.sbp.split(0)
        )
        self.deep_embedding = Embedding(
            deep_vocab_size, deep_embedding_vec_size // flow.env.get_world_size()
        )
        self.deep_embedding.to_global(
            flow.env.all_device_placement("cuda"), flow.sbp.split(1)
        )
        deep_feature_size = (
            deep_embedding_vec_size * num_deep_sparse_fields + num_dense_fields
        )
        self.linear_layers = nn.Sequential(
            OrderedDict(
                [
                    (
                        f"fc{i}",
                        Dense(
                            deep_feature_size if i == 0 else hidden_size,
                            hidden_size,
                            deep_dropout_rate,
                        ),
                    )
                    for i in range(hidden_units_num)
                ]
            )
        )
        self.linear_layers.to_global(
            flow.env.all_device_placement("cuda"), flow.sbp.broadcast
        )
        self.deep_scores = nn.Linear(hidden_size, 1)
        self.deep_scores.to_global(
            flow.env.all_device_placement("cuda"), flow.sbp.broadcast
        )
        self.sigmoid = nn.Sigmoid()
        self.sigmoid.to_global(
            flow.env.all_device_placement("cuda"), flow.sbp.broadcast
        )

    def forward(
        self, dense_fields, wide_sparse_fields, deep_sparse_fields
    ) -> flow.Tensor:
        wide_sparse_fields = wide_sparse_fields.to_global(sbp=flow.sbp.broadcast)
        wide_embedding = self.wide_embedding(wide_sparse_fields)
        wide_embedding = wide_embedding.view(
            -1, wide_embedding.shape[-1] * wide_embedding.shape[-2]
        )
        wide_scores = flow.sum(wide_embedding, dim=1, keepdim=True)
        wide_scores = wide_scores.to_global(
            sbp=flow.sbp.split(0), grad_sbp=flow.sbp.broadcast
        )
        deep_sparse_fields = deep_sparse_fields.to_global(sbp=flow.sbp.broadcast)
        deep_embedding = self.deep_embedding(deep_sparse_fields)
        deep_embedding = deep_embedding.to_global(
            sbp=flow.sbp.split(0), grad_sbp=flow.sbp.split(2)
        )
        deep_embedding = deep_embedding.view(
            -1, deep_embedding.shape[-1] * deep_embedding.shape[-2]
        )
        deep_features = flow.cat([deep_embedding, dense_fields], dim=1)
        deep_features = self.linear_layers(deep_features)
        deep_scores = self.deep_scores(deep_features)
        return self.sigmoid(wide_scores + deep_scores)


class LocalWideAndDeep(nn.Module):
    def __init__(
        self,
        wide_vocab_size: int,
        deep_vocab_size: int,
        deep_embedding_vec_size: int = 16,
        num_deep_sparse_fields: int = 26,
        num_dense_fields: int = 13,
        hidden_size: int = 1024,
        hidden_units_num: int = 7,
        deep_dropout_rate: float = 0.5,
    ):
        super(LocalWideAndDeep, self).__init__()
        self.wide_embedding = Embedding(wide_vocab_size, 1,)
        self.deep_embedding = Embedding(deep_vocab_size, deep_embedding_vec_size)
        deep_feature_size = (
            deep_embedding_vec_size * num_deep_sparse_fields + num_dense_fields
        )
        self.linear_layers = nn.Sequential(
            OrderedDict(
                [
                    (
                        f"fc{i}",
                        Dense(
                            deep_feature_size if i == 0 else hidden_size,
                            hidden_size,
                            deep_dropout_rate,
                        ),
                    )
                    for i in range(hidden_units_num)
                ]
            )
        )
        self.deep_scores = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(
        self, dense_fields, wide_sparse_fields, deep_sparse_fields
    ) -> flow.Tensor:
        wide_embedding = self.wide_embedding(wide_sparse_fields)
        wide_embedding = wide_embedding.view(
            -1, wide_embedding.shape[-1] * wide_embedding.shape[-2]
        )
        wide_scores = flow.sum(wide_embedding, dim=1, keepdim=True)
        deep_embedding = self.deep_embedding(deep_sparse_fields)
        deep_embedding = deep_embedding.view(
            -1, deep_embedding.shape[-1] * deep_embedding.shape[-2]
        )
        deep_features = flow.cat([deep_embedding, dense_fields], dim=1)
        deep_features = self.linear_layers(deep_features)
        deep_scores = self.deep_scores(deep_features)
        return self.sigmoid(wide_scores + deep_scores)


def make_wide_and_deep_module(args, is_global):
    if is_global:
        model = GlobalWideAndDeep(
            wide_vocab_size=args.wide_vocab_size,
            deep_vocab_size=args.deep_vocab_size,
            deep_embedding_vec_size=args.deep_embedding_vec_size,
            num_deep_sparse_fields=args.num_deep_sparse_fields,
            num_dense_fields=args.num_dense_fields,
            hidden_size=args.hidden_size,
            hidden_units_num=args.hidden_units_num,
            deep_dropout_rate=args.deep_dropout_rate,
        )
    else:
        model = LocalWideAndDeep(
            wide_vocab_size=args.wide_vocab_size,
            deep_vocab_size=args.deep_vocab_size,
            deep_embedding_vec_size=args.deep_embedding_vec_size,
            num_deep_sparse_fields=args.num_deep_sparse_fields,
            num_dense_fields=args.num_dense_fields,
            hidden_size=args.hidden_size,
            hidden_units_num=args.hidden_units_num,
            deep_dropout_rate=args.deep_dropout_rate,
        )
        model = model.to("cuda")
    return model
