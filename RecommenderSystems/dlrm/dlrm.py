import oneflow as flow
import oneflow.nn as nn
import numpy as np

from dataloader import num_dense_fields, num_sparse_fields

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
        num_embedding_fields,
        interaction_itself=False,
        interaction_padding=True,
    ):
        super(Interaction, self).__init__()
        self.interaction_itself = interaction_itself
        n_cols = num_embedding_fields + 2 if self.interaction_itself else num_embedding_fields + 1
        output_size = dense_feature_size + sum(range(n_cols))
        self.output_size = ((output_size + 8 - 1) // 8 * 8) if interaction_padding else output_size
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
    def __init__(
        self,
        embedding_vec_size,
        persistent_path,
        column_size_array,
        store_type,
        cache_memory_budget_mb_per_rank,
    ):
        assert column_size_array is not None
        vocab_size = sum(column_size_array)
        capacity_per_rank = (vocab_size // flow.env.get_world_size() + 16 -1 ) // 16 * 16

        scales = np.sqrt(1 / np.array(column_size_array))
        initializer_list = [
            {"initializer": {"type": "uniform", "low": -scale, "high": scale}} for scale in scales
        ]
        if store_type == "device_only":
            store_options = flow.one_embedding.make_device_mem_store_options(
                persistent_path, capacity_per_rank=capacity_per_rank,
            )
        elif store_type == "device_host":
            assert cache_memory_budget_mb_per_rank > 0
            store_options = flow.one_embedding.make_device_mem_cached_host_mem_store_options(
                persistent_path,
                device_memory_budget_mb_per_rank=cache_memory_budget_mb_per_rank,
                capacity_per_rank=capacity_per_rank,
            )
        elif store_type == "device_ssd":
            assert cache_memory_budget_mb_per_rank > 0
            store_options = flow.one_embedding.make_device_mem_cached_ssd_store_options(
                persistent_path, device_memory_budget_mb_per_rank=cache_memory_budget_mb_per_rank,
            )
        else:
            raise NotImplementedError("not support", store_type)

        super(OneEmbedding, self).__init__()
        self.one_embedding = flow.one_embedding.Embedding(
            "sparse_embedding",
            embedding_vec_size,
            flow.float,
            flow.int64,
            columns=initializer_list,
            store_options=store_options,
        )

    def forward(self, ids):
        return self.one_embedding.forward(ids)


class DLRMModule(nn.Module):
    def __init__(
        self,
        embedding_vec_size=128,
        bottom_mlp=[512, 256, 128],
        top_mlp=[1024, 1024, 512, 256],
        use_fusedmlp=True,
        persistent_path=None,
        column_size_array=None,
        one_embedding_store_type="device_host",
        cache_memory_budget_mb_per_rank=8192,
        interaction_itself=True,
        interaction_padding=True,
    ):
        super(DLRMModule, self).__init__()
        assert (
            embedding_vec_size == bottom_mlp[-1]
        ), "Embedding vector size must equle to bottom MLP output size"
        self.bottom_mlp = MLP(num_dense_fields, bottom_mlp, fused=use_fusedmlp)
        self.embedding = OneEmbedding(
            embedding_vec_size,
            persistent_path,
            column_size_array,
            one_embedding_store_type,
            cache_memory_budget_mb_per_rank,
        )
        self.interaction = Interaction(
            bottom_mlp[-1],
            num_sparse_fields,
            interaction_itself,
            interaction_padding=interaction_padding,
        )
        self.top_mlp = MLP(
            self.interaction.output_size,
            top_mlp + [1],
            skip_final_activation=True,
            fused=use_fusedmlp,
        )

    def forward(self, dense_fields, sparse_fields) -> flow.Tensor:
        dense_fields = flow.log(dense_fields + 1.0)
        dense_fields = self.bottom_mlp(dense_fields)
        embedding = self.embedding(sparse_fields)
        features = self.interaction(dense_fields, embedding)
        return self.top_mlp(features)


def make_dlrm_module(args):
    model = DLRMModule(
        embedding_vec_size=args.embedding_vec_size,
        bottom_mlp=args.bottom_mlp,
        top_mlp=args.top_mlp,
        use_fusedmlp=args.use_fusedmlp,
        persistent_path=args.persistent_path,
        column_size_array=args.column_size_array,
        one_embedding_store_type=args.store_type,
        cache_memory_budget_mb_per_rank=args.cache_memory_budget_mb_per_rank,
        interaction_itself=args.interaction_itself,
        interaction_padding=not args.disable_interaction_padding,
    )
    return model
