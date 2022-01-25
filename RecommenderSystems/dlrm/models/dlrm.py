from collections import OrderedDict
import oneflow as flow
from oneflow.framework.tensor import _xor
import oneflow.nn as nn
from typing import Any
import numpy as np
import os

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
                nn.init.normal_(param, 0.0, np.sqrt(2 / (in_features + out_features)))
            elif name.endswith("bias"):
                nn.init.normal_(param, 0.0, np.sqrt(1 / out_features))

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
    def __init__(self, interaction_type='dot', interaction_itself=False, num_sparse_fields=26):
        super(Interaction, self).__init__()
        self.interaction_type = interaction_type
        self.interaction_itself = interaction_itself
        self.num_sparse_fields = num_sparse_fields

        # slice
        offset = 1 if self.interaction_itself else 0
        # indices = flow.tensor([i * 27 + j for i in range(27) for j in range(i + offset)])
        # self.register_buffer("indices", indices)
        li = flow.tensor([i for i in range(27) for j in range(i + offset)])
        lj = flow.tensor([j for i in range(27) for j in range(i + offset)])
        self.register_buffer("li", li)
        self.register_buffer("lj", lj)
        
    def forward(self, x:flow.Tensor, ly:flow.Tensor) -> flow.Tensor:
        # x - dense fields, ly = embedding
        if self.interaction_type == 'cat':
            R = flow.cat([x, ly], dim=1)
        elif self.interaction_type == 'dot': # slice
            (batch_size, d) = x.shape
            T = flow.cat([x, ly], dim=1).view((batch_size, -1, d))
            # perform a dot product
            Z = flow.bmm(T, flow.transpose(T, 1, 2))
            Zflat = Z[:, self.li, self.lj]
            R = flow.cat([x, Zflat], dim=1)       
        else:
            assert 0, 'dot or cat'
        return R

    def output_feature_size(self, embedding_vec_size, dense_feature_size):
        if self.interaction_type == 'dot':
            # assert 0, 'dot is not supported yet'
            assert embedding_vec_size == dense_feature_size, "Embedding vector size must equle to dense feature size"
            n_cols = self.num_sparse_fields + 1
            if self.interaction_itself:
                n_cols += 1
            return dense_feature_size + sum(range(n_cols))
        elif self.interaction_type == 'cat':
            return embedding_vec_size * self.num_sparse_fields + dense_feature_size
        else:
            assert 0, 'dot or cat'


class Embedding(nn.Embedding):
    def __init__(self, vocab_size, embed_size):
        super(Embedding, self).__init__(vocab_size, embed_size, padding_idx=0)
        for param in self.parameters():
            nn.init.uniform_(param, a=-0.05, b=0.05)
            # W = np.load('/tank/model_zoo/dlrm_baseline_params_emb16/embedding_weight.npy')
            # param.data = flow.tensor(W, requires_grad=True)

slot_size_array = np.array(
    [
        227605432,
        39060,
        17295,
        7424,
        20265,
        3,
        7122,
        1543,
        63,
        130229467,
        3067956,
        405282,
        10,
        2209,
        11938,
        155,
        4,
        976,
        14,
        292775614,
        40790948,
        187188510,
        590152,
        12973,
        108,
        36,
    ]
)
scales = np.sqrt(1 / slot_size_array)
initializer_list = []
for i in range(scales.size):
    initializer_list.append(
        {"initializer": {"type": "uniform", "low": -scales[i], "high": scales[i],}}
    )


class OneEmbedding(nn.OneEmbeddingLookup):
    def __init__(self, vocab_size, embed_size):
        options = {
            "dtype": flow.float,
            "name": "my_embedding",
            "embedding_dim": embed_size,
            "cache" : [
                {
                    "policy": "lru",
                    "cache_memory_budget_mb": 16384,
                    "value_memory_kind": "device",
                },
                #{
                #    "policy": "full",
                #    "cache_memory_budget_mb": 163840,
                #    "value_memory_kind": "host",
                #}
            ],
            "kv_store": {
                "persistent_table": {
                    "path": os.environ.get("BLOCK_BASED_PATH"),
                    "physical_block_size": 512,
                },
            },
            "default_initializer": {"type": "normal", "mean": 0, "std": 1},
            "columns": initializer_list,
            "optimizer": {
                "lr": {
                    "base_lr": 24,
                    "decay": {
                        "type": "polynomial",
                        "decay_batches": 27772,
                        "end_lr": 0.0,
                        "power": 2.0,
                        "cycle": False,
                    },
                    "warmup": {
                        "type": "linear",
                        "warmup_batches": 2750,
                        "start_multiplier": 0.0,
                    },
                },
                "type": "sgd",
                "momentum": 0.0,
                "betas": [0.9, 0.999],
                "eps": 1e-8,
            },
        }
        super(OneEmbedding, self).__init__(options)
        column_id = flow.tensor(range(26), dtype=flow.int32).reshape(1,26)
        self.register_buffer("column_id", column_id)

    def forward(self, ids):
        bsz = ids.shape[0]
        column_id = flow.ones((bsz, 1), dtype=flow.int32, sbp=ids.sbp, placement=ids.placement) * self.column_id
        if (ids.is_consistent):
            column_id = column_id.to_consistent(sbp=ids.sbp, placement=ids.placement)
        return super(OneEmbedding, self._origin).forward(ids, column_id)


embd_dict = {
    'OneEmbedding': OneEmbedding,
    'Embedding': Embedding,
}


class DLRMModule(nn.Module):
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
        embedding_type = "Embedding",
    ):
        super(DLRMModule, self).__init__()
        self.bottom_mlp = MLP(num_dense_fields, bottom_mlp)
        self.embedding = embd_dict[embedding_type](vocab_size, embedding_vec_size)
        self.interaction = Interaction(interaction_type, interaction_itself, num_sparse_fields)
        feature_size = self.interaction.output_feature_size(embedding_vec_size, bottom_mlp[-1])        
        self.top_mlp = MLP(feature_size, top_mlp)
        self.scores = nn.Linear(top_mlp[-1], 1)
        self.sigmoid = nn.Sigmoid()


    def forward(self, dense_fields, sparse_fields) -> flow.Tensor:
        dense_fields = flow.log(dense_fields + 2.0)
        dense_fields = self.bottom_mlp(dense_fields)
        sparse_fields = flow.cast(sparse_fields, flow.int64)
        embedding = self.embedding(sparse_fields)
        embedding = embedding.view(-1, embedding.shape[-1] * embedding.shape[-2])
        features = self.interaction(dense_fields, embedding)
        features = self.top_mlp(features)
        scores = self.scores(features)
        return self.sigmoid(scores)


def make_dlrm_module(args):
    model = DLRMModule(
        vocab_size=args.vocab_size,
        embedding_vec_size=args.embedding_vec_size,
        num_sparse_fields=args.num_sparse_fields,
        num_dense_fields=args.num_dense_fields,
        bottom_mlp=args.bottom_mlp,
        top_mlp=args.top_mlp,
        interaction_type = args.interaction_type,
        interaction_itself = args.interaction_itself,
        embedding_type = args.embedding_type,
    )
    return model
