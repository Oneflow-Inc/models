import sys
from collections import OrderedDict
import oneflow as flow
import oneflow.nn as nn
import numpy as np


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


class OneEmbedding(nn.Module):
    def __init__(self, args):
        assert args.column_size_array is not None
        scales = np.sqrt(1 / np.array(args.column_size_array))
        initializer_list = []
        for i in range(scales.size):
            initializer_list.append(
                {
                    "initializer": {
                        "type": "uniform",
                        "low": -scales[i],
                        "high": scales[i],
                    }
                }
            )
        if args.store_type == "device_only":
            store_options = flow.one_embedding.make_device_mem_store_options(
                device_memory_budget_mb_per_rank=args.cache_memory_budget_mb[0],
                persistent_path=args.persistent_path,
                size_factor=1,
            )
        elif args.store_type == "host_only":
            store_options = flow.one_embedding.make_host_mem_store_options(
                host_memory_budget_mb_per_rank=args.cache_memory_budget_mb[0],
                persistent_path=args.persistent_path,
                size_factor=1,
            )
        elif args.store_type == "device_ssd":
            store_options = flow.one_embedding.make_device_mem_cached_ssd_store_options(
                device_memory_budget_mb_per_rank=args.cache_memory_budget_mb[0],
                persistent_path=args.persistent_path,
                size_factor=1,
            )
        elif args.store_type == "host_ssd":
            store_options = flow.one_embedding.make_host_mem_cached_ssd_store_options(
                host_memory_budget_mb_per_rank=args.cache_memory_budget_mb[0],
                persistent_path=args.persistent_path,
                size_factor=1,
            )
        elif args.store_type == "device_host":
            store_options = flow.one_embedding.make_device_mem_cached_host_mem_store_options(
                device_memory_budget_mb_per_rank=args.cache_memory_budget_mb[0],
                host_memory_budget_mb_per_rank=args.cache_memory_budget_mb[1],
                persistent_path=args.persistent_path,
                size_factor=1,
            )
        else:
            raise NotImplementedError("not support", args.store_type)
        print("store_options", store_options)

        super(OneEmbedding, self).__init__()
        self.one_embedding = flow.one_embedding.Embedding(
            "my_embedding",
            args.embedding_vec_size,
            flow.float,
            flow.int64,
            columns=initializer_list,
            store_options=store_options,
        )

    def forward(self, ids):
        return self.one_embedding.forward(ids)


def make_mlp_module(mlp_type):
    if mlp_type.lower() == "mlp":
        return MLP
    elif mlp_type.lower() == "fusedmlp":
        return FusedMLP
    else:
        assert f"Unsupported MLP type: {mlp_type}"


class DLRMModule(nn.Module):
    def __init__(self, args):
        super(DLRMModule, self).__init__()
        self.bottom_mlp = make_mlp_module(args.mlp_type)(args.num_dense_fields, args.bottom_mlp)
        self.embedding = OneEmbedding(args)
        self.interaction = Interaction(args.interaction_itself, args.num_sparse_fields,
                                       args.output_padding)
        feature_size = self.interaction.output_feature_size(args.embedding_vec_size,
                                                            args.bottom_mlp[-1])
        self.top_mlp = make_mlp_module(args.mlp_type)(feature_size, args.top_mlp + [1],
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
