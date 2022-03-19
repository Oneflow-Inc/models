from collections import OrderedDict
import oneflow as flow
import oneflow.nn as nn
import numpy as np


__all__ = ["make_dlrm_module"]


class Dense(nn.Module):
    def __init__(self, in_features: int, out_features: int, relu=True) -> None:
        super(Dense, self).__init__()
        self.features = (
            nn.Sequential(nn.Linear(in_features, out_features), nn.ReLU(inplace=True))
            if relu
            else nn.Linear(in_features, out_features)
        )

    def forward(self, x: flow.Tensor) -> flow.Tensor:
        return self.features(x)


class MLP(nn.Module):
    def __init__(
        self, in_features: int, hidden_units, skip_final_activation=False, fused=True
    ) -> None:
        super(MLP, self).__init__()
        if fused:
            self.linear_layers = nn.FusedMLP(
                in_features,
                hidden_units[:-1],
                hidden_units[-1],
                skip_final_activation=skip_final_activation,
            )
        else:
            units = [in_features] + hidden_units
            num_layers = len(hidden_units)
            denses = [
                Dense(units[i], units[i + 1], not skip_final_activation or (i + 1) < num_layers)
                for i in range(num_layers)
            ]
            self.linear_layers = nn.Sequential(*denses)

        for name, param in self.linear_layers.named_parameters():
            if "weight" in name:
                nn.init.normal_(param, 0.0, np.sqrt(2 / sum(param.shape)))
            elif "bias" in name:
                nn.init.normal_(param, 0.0, np.sqrt(1 / param.shape[0]))

    def forward(self, x: flow.Tensor) -> flow.Tensor:
        return self.linear_layers(x)


class Interaction(nn.Module):
    def __init__(
        self,
        dense_feature_size,
        num_sparse_fields=26,
        interaction_itself=False,
        interaction_padding=True,
    ):
        super(Interaction, self).__init__()
        self.interaction_itself = interaction_itself
        n_cols = num_sparse_fields + 2 if self.interaction_itself else num_sparse_fields + 1
        output_size = dense_feature_size + sum(range(n_cols))
        self.output_size = ((output_size + 8 - 1) & (-8)) if interaction_padding else output_size
        self.output_padding = self.output_size - output_size

    def forward(self, x: flow.Tensor, ly: flow.Tensor) -> flow.Tensor:
        # x - dense fields, ly = embedding
        (bsz, d) = x.shape
        return flow._C.fused_dot_feature_interaction(
            [x.view(bsz, 1, d), ly],
            output_concat=x,
            self_interaction=self.interaction_itself,
            output_padding=self.output_padding,
        )


class OneEmbedding(nn.Module):
    def __init__(self, args):
        assert args.column_size_array is not None
        vocab_size = sum(args.column_size_array)
        capacity_per_rank = (vocab_size // flow.env.get_world_size() + 15) & (-16)

        scales = np.sqrt(1 / np.array(args.column_size_array))
        initializer_list = [
            {"initializer": {"type": "uniform", "low": -scales[i], "high": scales[i],}}
            for i in range(scales.size)
        ]
        if args.store_type == "device_only":
            store_options = flow.one_embedding.make_device_mem_store_options(
                args.persistent_path, capacity_per_rank=capacity_per_rank, size_factor=1,
            )
        elif args.store_type == "device_host":
            assert args.device_memory_budget_mb_per_rank > 0
            store_options = flow.one_embedding.make_device_mem_cached_host_mem_store_options(
                args.persistent_path,
                device_memory_budget_mb_per_rank=args.device_memory_budget_mb_per_rank,
                capacity_per_rank=capacity_per_rank,
                size_factor=1,
            )
        elif args.store_type == "device_ssd":
            assert args.device_memory_budget_mb_per_rank > 0
            store_options = flow.one_embedding.make_device_mem_cached_ssd_store_options(
                args.persistent_path,
                device_memory_budget_mb_per_rank=args.device_memory_budget_mb_per_rank,
                size_factor=1,
            )
        else:
            raise NotImplementedError("not support", args.store_type)

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


class DLRMModule(nn.Module):
    def __init__(self, args):
        super(DLRMModule, self).__init__()
        assert (
            args.embedding_vec_size == args.bottom_mlp[-1]
        ), "Embedding vector size must equle to bottom MLP output size"
        self.bottom_mlp = MLP(args.num_dense_fields, args.bottom_mlp, fused=args.enable_fusedmlp)
        self.embedding = OneEmbedding(args)
        self.interaction = Interaction(
            args.bottom_mlp[-1],
            args.num_sparse_fields,
            args.interaction_itself,
            interaction_padding=(not args.disable_interaction_padding),
        )
        self.top_mlp = MLP(
            self.interaction.output_size,
            args.top_mlp + [1],
            skip_final_activation=True,
            fused=args.enable_fusedmlp,
        )

    def forward(self, dense_fields, sparse_fields) -> flow.Tensor:
        dense_fields = flow.log(dense_fields + 1.0)
        dense_fields = self.bottom_mlp(dense_fields)
        embedding = self.embedding(sparse_fields)
        features = self.interaction(dense_fields, embedding)
        return self.top_mlp(features)


def make_dlrm_module(args):
    model = DLRMModule(args)
    return model
