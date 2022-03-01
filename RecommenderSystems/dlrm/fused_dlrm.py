import oneflow as flow
import oneflow.nn as nn
import numpy as np

__all__ = ["make_dlrm_module"]


class MLP(nn.Module):
    def __init__(self, in_features: int, hidden_units, skip_final_activation=False) -> None:
        super(MLP, self).__init__()
        self.linear_layers = nn.FusedMLP(in_features, hidden_units[:-1], hidden_units[-1], 
                                         skip_final_activation=skip_final_activation)
        for name, param in self.linear_layers.named_parameters():
            idx = int(name.split("_")[1])
            if name.startswith("weight"):
                nn.init.normal_(param, 0.0, np.sqrt(2 / (in_features + hidden_units[0]) if idx == 0 else np.sqrt(2 / (hidden_units[idx-1] + hidden_units[idx]))))
            elif name.startswith("bias"):
                nn.init.normal_(param, 0.0, np.sqrt(2 / (hidden_units[idx])))

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
    def __init__(self, embed_size, args):
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
            "embedding_dim": embed_size,
            "storage_dim": embed_size,
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
        if (ids.is_global):
            column_id = column_id.to_global(sbp=ids.sbp, placement=ids.placement)
        return self.one_embedding.forward(ids, column_id)

    def set_model_parallel(self, placement=None):
        pass


class DLRMModule(nn.Module):
    def __init__(self, args):
        super(DLRMModule, self).__init__()
        self.bottom_mlp = MLP(args.num_dense_fields, args.bottom_mlp)
        self.embedding = OneEmbedding(args.embedding_vec_size, args)
        self.interaction = Interaction(args.interaction_itself, args.num_sparse_fields)
        feature_size = self.interaction.output_feature_size(args.embedding_vec_size, args.bottom_mlp[-1])        
        self.top_mlp = MLP(feature_size, args.top_mlp + [1], skip_final_activation=True)
        # self.scores = MLP(args.top_mlp[-1], [1], skip_final_activation=True)

    def forward(self, dense_fields, sparse_fields) -> flow.Tensor:
        dense_fields = flow.log(dense_fields + 1.0)
        dense_fields = self.bottom_mlp(dense_fields)
        embedding = self.embedding(sparse_fields)
        features = self.interaction(dense_fields, embedding)
        return self.top_mlp(features)
        # features = self.top_mlp(features)
        # scores = self.scores(features)
        # return scores

def make_dlrm_module(args):
    model = DLRMModule(args)
    return model
