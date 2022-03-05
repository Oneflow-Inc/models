import sys
from collections import OrderedDict
import oneflow as flow
import oneflow.nn as nn
import numpy as np


def NameToClass(classname):
    return getattr(sys.modules[__name__], classname)


__all__ = ["make_dlrm_module"]


class Dense(nn.Module):
    def __init__(self, in_features: int, out_features: int, relu=True) -> None:
        super(Dense, self).__init__()
        self.features = nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.ReLU(inplace=True),
        ) if relu else nn.Linear(in_features, out_features)
        for name, param in self.named_parameters():
            if name.endswith("weight"):
                nn.init.normal_(param, 0.0, np.sqrt(2 / (in_features + out_features)))
            elif name.endswith("bias"):
                nn.init.normal_(param, 0.0, np.sqrt(1 / out_features))

    def forward(self, x: flow.Tensor) -> flow.Tensor:
        x = self.features(x)
        return x


class MLP(nn.Module):
    def __init__(self, in_features: int, hidden_units, skip_final_activation=False) -> None:
        super(MLP, self).__init__()
        print('using MLP')
        units = [in_features] + hidden_units
        num_layers = len(hidden_units)
        self.linear_layers = nn.Sequential(
            OrderedDict(
                [
                    (
                        f"fc{i}",
                        Dense(units[i], units[i+1], not skip_final_activation or (i+1)<num_layers)
                    )
                    for i in range(num_layers)
                ]
            )
        )
    def forward(self, x:flow.Tensor) -> flow.Tensor:
        return self.linear_layers(x)


class FusedMLP(nn.Module):
    def __init__(self, in_features: int, hidden_units, skip_final_activation=False) -> None:
        super(FusedMLP, self).__init__()
        print('using fusedMLP')
        self.linear_layers = nn.FusedMLP(in_features, hidden_units[:-1], hidden_units[-1], 
                                         skip_final_activation=skip_final_activation)
        units = [in_features] + hidden_units
        w_init_factor = [units[i] + units[i + 1] for i in range(len(hidden_units))]
        for name, param in self.linear_layers.named_parameters():
            idx = int(name.split("_")[1])
            if name.startswith("weight"):
                nn.init.normal_(param, 0.0, np.sqrt(2 / w_init_factor[idx]))
            elif name.startswith("bias"):
                nn.init.normal_(param, 0.0, np.sqrt(1 / hidden_units[idx]))

    def forward(self, x:flow.Tensor) -> flow.Tensor:
        return self.linear_layers(x)


class Interaction(nn.Module):
    def __init__(self, interaction_itself=False, num_sparse_fields=26, output_padding=1):
        super(Interaction, self).__init__()
        self.interaction_itself = interaction_itself
        self.num_sparse_fields = num_sparse_fields
        self.output_padding = output_padding

    def forward(self, x:flow.Tensor, ly:flow.Tensor) -> flow.Tensor:
        # x - dense fields, ly = embedding
        (bsz, d) = x.shape
        return flow._C.fused_dot_feature_interaction(
            [x.view(bsz, 1, d), ly],
            output_concat=x, 
            self_interaction=self.interaction_itself, 
            output_padding=self.output_padding
        )

    def output_feature_size(self, embedding_vec_size, dense_feature_size):
        assert embedding_vec_size == dense_feature_size, "Embedding vector size must equle to dense feature size"
        n_cols = self.num_sparse_fields + 1
        if self.interaction_itself:
            n_cols += 1
        return dense_feature_size + sum(range(n_cols)) + self.output_padding


class Embedding(nn.Embedding):
    def __init__(self, args):
        assert args.embedding_split_axis < 2, "Embedding model parallel split axis is 0 or 1."
        self.split_axis = args.embedding_split_axis
        super(Embedding, self).__init__(args.vocab_size, args.embedding_vec_size, padding_idx=0)
        for param in self.parameters():
            nn.init.uniform_(param, a=-0.05, b=0.05)

    def set_model_parallel(self, placement=None):
        # Overriding to_global function does not work
        # because to_global call is not recursive
        if self.split_axis >= 0:
            self.to_global(placement, flow.sbp.split(self.split_axis))

    def forward(self, ids):
        if self.split_axis >= 0:
            ids = ids.to_global(sbp=flow.sbp.broadcast)
        
        # Forward
        # weight    ids => embedding
        # S(0)      B   => P
        # S(1)      B   => S(2)
        embeddings = flow._C.gather(self.weight, ids, axis=0)
        # Backward: unsorted_segment_sum_like
        # segment_ids   data            like    => out
        # ids           embedding_grad  weight  => weight_grad
        # B             B               S(0)    => S(0)
        # B             S(2)            S(1)    => S(1)

        if self.split_axis == 0:
            # Forward: P => S(0), Backward: S(0) => B
            return embeddings.to_global(sbp=flow.sbp.split(0), grad_sbp=flow.sbp.broadcast)
        elif self.split_axis == 1:
            # Forward: S(2) => S(0), Backward: S(0) => S(2)
            return embeddings.to_global(sbp=flow.sbp.split(0), grad_sbp=flow.sbp.split(2))
        else:
            return embeddings


class OneEmbedding(nn.Module):
    def __init__(self, args):
        assert args.column_size_array is not None
        scales = np.sqrt(1 / np.array(args.column_size_array))
        initializer_list = []
        for i in range(scales.size):
            initializer_list.append(
                {"initializer": {"type": "uniform", "low": -scales[i], "high": scales[i],}}
            )
        cache_list = []
        assert len(args.cache_policy) <= 2
        assert len(args.cache_policy) == len(args.cache_memory_budget_mb)
        assert len(args.cache_policy) == len(args.value_memory_kind)
        for i in range(len(args.cache_policy)):
            if args.cache_policy[i] != "none":
                cache = {
                    "policy": args.cache_policy[i],
                    "cache_memory_budget_mb": args.cache_memory_budget_mb[i],
                    "value_memory_kind": args.value_memory_kind[i]
                }
                cache_list.append(cache)
        print("cache_list", cache_list)
        options = {
            "key_type": flow.int64,
            "value_type": flow.float,
            "name": "my_embedding",
            "embedding_dim": args.embedding_vec_size,
            "storage_dim": args.embedding_vec_size,
            "kv_store": {
                "caches" : cache_list,
                "persistent_table": {
                    "path": args.persistent_path,
                    "physical_block_size": 512,
                },
            },
            "default_initializer": {"type": "normal", "mean": 0, "std": 0.05},
            "columns": initializer_list,
        }
        super(OneEmbedding, self).__init__()
        column_id = flow.tensor(range(26), dtype=flow.int32).reshape(1,26)
        self.register_buffer("column_id", column_id)
        self.one_embedding = nn.OneEmbeddingLookup(options)

    def forward(self, ids):
        bsz = ids.shape[0]
        column_id = flow.ones((bsz, 1), dtype=flow.int32, sbp=ids.sbp, placement=ids.placement) * self.column_id
        column_id = column_id.to_global(sbp=ids.sbp, placement=ids.placement)
        return self.one_embedding.forward(ids, column_id)

    def set_model_parallel(self, placement=None):
        pass


embd_dict = {
    'OneEmbedding': OneEmbedding,
    'Embedding': Embedding,
}


class DLRMModule(nn.Module):
    def __init__(self, args):
        super(DLRMModule, self).__init__()
        self.bottom_mlp = NameToClass(args.mlp_type)(args.num_dense_fields, args.bottom_mlp)
        self.embedding = NameToClass(args.embedding_type)(args)
        self.interaction = Interaction(args.interaction_itself, args.num_sparse_fields,
                                       args.output_padding)
        feature_size = self.interaction.output_feature_size(args.embedding_vec_size,
                                                            args.bottom_mlp[-1])
        self.top_mlp = NameToClass(args.mlp_type)(feature_size, args.top_mlp + [1],
                                                  skip_final_activation=True)

    def forward(self, dense_fields, sparse_fields) -> flow.Tensor:
        dense_fields = flow.log(dense_fields + 1.0)
        dense_fields = self.bottom_mlp(dense_fields)
        embedding = self.embedding(sparse_fields)
        features = self.interaction(dense_fields, embedding)
        return self.top_mlp(features)

def make_dlrm_module(args):
    model = DLRMModule(args)
    return model
