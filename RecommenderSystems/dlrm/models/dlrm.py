from collections import OrderedDict
import oneflow as flow
from oneflow.framework.tensor import _xor
import oneflow.nn as nn
from typing import Any


__all__ = ["make_dlrm_module"]


class Dense(nn.Module):
    def __init__(self, in_features: int, out_features: int) -> None:
        super(Dense, self).__init__()
        self.features = nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.ReLU(inplace=True),
        )
        for name, param in self.named_parameters():
            if name.endswith("weight"):
                nn.init.xavier_uniform_(param)
            elif name.endswith("bias"):
                nn.init.zeros_(param)

    def forward(self, x: flow.Tensor) -> flow.Tensor:
        x = self.features(x)
        return x


class MLP(nn.Module):
    def __init__(self, in_features: int, hidden_units) -> None:
        super(MLP, self).__init__()
        self.linear_layers = nn.Sequential(
            OrderedDict(
                [
                    (
                        f"fc{i}",
                        Dense(in_features if i == 0 else hidden_units[i-1], h)
                    )
                    for i, h in enumerate(hidden_units)
                ]
            )
        )
    def forward(self, x:flow.Tensor) -> flow.Tensor:
        return self.linear_layers(x)


class Interaction(nn.Module):
    def __init__(self, interaction_type='dot', interaction_itself=False):
        super(Interaction, self).__init__()
        self.interaction_type = interaction_type
        self.interaction_itself = interaction_itself

        # offset = 1 if self.interaction_itself else 0
        # self.li = flow.tensor([i for i in range(27) for j in range(i + offset)])
        # self.lj = flow.tensor([j for i in range(27) for j in range(i + offset)])

    def forward(self, x:flow.Tensor, ly:flow.Tensor) -> flow.Tensor:
        # x - dense fields, ly = embedding
        if self.interaction_type == 'dot':
            (batch_size, d) = x.shape
            T = flow.cat([x, ly], dim=1).view((batch_size, -1, d))
            # perform a dot product
            Z = flow.bmm(T, flow.transpose(T, 1, 2))
            _, ni, nj = Z.shape
            offset = 1 if self.interaction_itself else 0
            li = flow.tensor([i for i in range(ni) for j in range(i + offset)])
            lj = flow.tensor([j for i in range(nj) for j in range(i + offset)])
            Zflat = Z[:, li, lj]
            # Zflat = Z[:, self.li, self.lj]
            # concatenate dense features and interactions
            R = flow.cat([x, Zflat], dim=1)
        elif self.interaction_type == 'cat':
            R = flow.cat([x, ly], dim=1)
        else:
            assert 0, 'dot or cat'
        return R

    def output_feature_size(self, num_sparse_fields, embedding_vec_size, dense_feature_size):
        if self.interaction_type == 'dot':
            # assert 0, 'dot is not supported yet'
            assert embedding_vec_size == dense_feature_size, "Embedding vector size must equle to dense feature size"
            n_cols = num_sparse_fields + 1
            if self.interaction_itself:
                n_cols += 1
            return dense_feature_size + sum(range(n_cols))
        elif self.interaction_type == 'cat':
            return embedding_vec_size * num_sparse_fields + dense_feature_size
        else:
            assert 0, 'dot or cat'


class Embedding(nn.Embedding):
    def __init__(self, vocab_size, embed_size):
        super(Embedding, self).__init__(vocab_size, embed_size, padding_idx=0)
        for param in self.parameters():
            nn.init.uniform_(param, a=-0.05, b=0.05)


# class Embedding(nn.OneEmbeddingLookup):
#     def __init__(self, vocab_size, embed_size):
#         options = {
#             "name": "my_embedding",
#             # Can't change the embedding_size 128 because the kv store value_length has been set to 128
#             "embedding_size": 128,
#             "dtype": flow.float,
#             "encoder": "invalid",
#             "partitioning": "invalid",
#             "initializer": "invalid",
#             "optimizer": "invalid",
#             "backend": "invalid",
#         }
#         super(Embedding, self).__init__(options)

class ConsistentDLRM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embedding_vec_size: int = 16,
        num_sparse_fields: int = 26,
        num_dense_fields: int = 13,
        bottom_mlp = [],
        top_mlp = [],
        interaction_type = 'dot',
        interaction_itself = False,
    ):
        super(ConsistentDLRM, self).__init__()
        self.bottom_mlp = MLP(num_dense_fields, bottom_mlp)
        self.bottom_mlp.to_consistent(flow.env.all_device_placement("cuda"), flow.sbp.broadcast)

        self.embedding = Embedding(vocab_size, embedding_vec_size // flow.env.get_world_size())
        self.embedding.to_consistent(flow.env.all_device_placement("cuda"), flow.sbp.split(0))
        self.interaction = Interaction(interaction_type, interaction_itself)
        self.interaction.to_consistent(flow.env.all_device_placement("cuda"), flow.sbp.broadcast)
        # self.interaction.li.to_consistent(flow.env.all_device_placement("cuda"), flow.sbp.broadcast)
        # self.interaction.lj.to_consistent(flow.env.all_device_placement("cuda"), flow.sbp.broadcast)
        feature_size = self.interaction.output_feature_size(num_sparse_fields, embedding_vec_size,
                                                            bottom_mlp[-1])
        self.top_mlp = MLP(feature_size, top_mlp)
        self.top_mlp.to_consistent(flow.env.all_device_placement("cuda"), flow.sbp.broadcast)
        self.scores = nn.Linear(top_mlp[-1], 1)
        self.scores.to_consistent(flow.env.all_device_placement("cuda"), flow.sbp.broadcast)
        self.sigmoid = nn.Sigmoid()
        self.sigmoid.to_consistent(flow.env.all_device_placement("cuda"), flow.sbp.broadcast)

    def forward(self, dense_fields, sparse_fields) -> flow.Tensor:
        dense_fields = dense_fields.to_consistent(sbp=flow.sbp.split(0))
        dense_fields = self.bottom_mlp(dense_fields)

        sparse_fields = sparse_fields.to_consistent(sbp=flow.sbp.split(0))
        sparse_fields = flow.cast(sparse_fields, flow.int64)
        embedding = self.embedding(sparse_fields)
        embedding = embedding.view(-1, embedding.shape[-1] * embedding.shape[-2])
        features = self.interaction(dense_fields, embedding)
        features = self.top_mlp(features)
        scores = self.scores(features)
        return self.sigmoid(scores)


class LocalDLRM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embedding_vec_size: int = 16,
        num_sparse_fields: int = 26,
        num_dense_fields: int = 13,
        bottom_mlp = [],
        top_mlp = [],
        interaction_type = 'dot',
        interaction_itself = False,
    ):
        super(LocalDLRM, self).__init__()
        self.bottom_mlp = MLP(num_dense_fields, bottom_mlp)
        self.embedding = Embedding(vocab_size, embedding_vec_size)
        self.interaction = Interaction(interaction_type, interaction_itself)
        feature_size = self.interaction.output_feature_size(num_sparse_fields, embedding_vec_size,
                                                            bottom_mlp[-1])        
        self.top_mlp = MLP(feature_size, top_mlp)
        self.scores = nn.Linear(top_mlp[-1], 1)
        self.sigmoid = nn.Sigmoid()


    def forward(self, dense_fields, sparse_fields) -> flow.Tensor:
        dense_fields = self.bottom_mlp(dense_fields)
        sparse_fields = flow.cast(sparse_fields, flow.int64)
        embedding = self.embedding(sparse_fields)
        embedding = embedding.view(-1, embedding.shape[-1] * embedding.shape[-2])
        # features = flow.cat([embedding, dense_fields], dim=1)
        features = self.interaction(dense_fields, embedding)
        features = self.top_mlp(features)
        scores = self.scores(features)
        return self.sigmoid(scores)


def make_dlrm_module(args, is_consistent):
    if is_consistent:
        model = ConsistentDLRM(
            vocab_size=args.vocab_size,
            embedding_vec_size=args.embedding_vec_size,
            num_sparse_fields=args.num_sparse_fields,
            num_dense_fields=args.num_dense_fields,
            bottom_mlp=args.bottom_mlp,
            top_mlp=args.top_mlp,
            interaction_type = args.interaction_type,
            interaction_itself = args.interaction_itself,
        )
    else:
        model = LocalDLRM(
            vocab_size=args.vocab_size,
            embedding_vec_size=args.embedding_vec_size,
            num_sparse_fields=args.num_sparse_fields,
            num_dense_fields=args.num_dense_fields,
            bottom_mlp=args.bottom_mlp,
            top_mlp=args.top_mlp,
            interaction_type = args.interaction_type,
            interaction_itself = args.interaction_itself,
        )
        model = model.to("cuda")
    return model
