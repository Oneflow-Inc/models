from collections import OrderedDict
import oneflow as flow
from oneflow.framework.tensor import _xor
import oneflow.nn as nn
from typing import Any
import numpy as np
import os

__all__ = ["make_dlrm_module"]


class MLP(nn.Module):
    def __init__(self, in_features: int, hidden_units, skip_final_activation=False) -> None:
        super(MLP, self).__init__()
        self.linear_layers = nn.FusedMLP(in_features, hidden_units[:-1], hidden_units[-1], skip_final_activation=skip_final_activation)
        self.layer_num = len(hidden_units)
        self.w_init_factor = [in_features + hidden_units[0]]
        self.b_init_factor = [hidden_units[0]]
        for idx in range(1, self.layer_num): 
            self.w_init_factor.append(hidden_units[idx-1] + hidden_units[idx])
            self.b_init_factor.append(hidden_units[idx])

        for name, param in self.linear_layers.named_parameters():
            idx = int(name.split("_")[1])
            if name.startswith("weight"):
                nn.init.normal_(param, 0.0, np.sqrt(2 / self.w_init_factor[idx]))
            elif name.startswith("bias"):
                nn.init.normal_(param, 0.0, np.sqrt(2 / self.b_init_factor[idx]))

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
        if interaction_type == 'dot':
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
            Z = flow.matmul(T, T, transpose_b=True)
            Zflat = Z[:, self.li, self.lj]
            R = flow.cat([x, Zflat], dim=1)
        elif self.interaction_type == 'fused':
            (batch_size, d) = x.shape
            R = flow._C.fused_dot_feature_interaction([flow.reshape(x,(batch_size,1,d)), flow.reshape(ly, (batch_size, -1,d))], output_concat=x, self_interaction=False, output_padding=1)
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
        elif self.interaction_type == 'fused':
            assert embedding_vec_size == dense_feature_size, "Embedding vector size must equle to dense feature size"
            n_cols = self.num_sparse_fields + 1
            if self.interaction_itself:
                n_cols += 1
            return dense_feature_size + sum(range(n_cols)) + 1
        else:
            assert 0, 'dot or cat'


class Embedding(nn.Embedding):
    def __init__(self, vocab_size, embed_size, args):
        if args.is_global:
            assert args.embedding_split_axis < 2, "Embedding model parallel split axis can only be 0 or 1."
            self.split_axis = args.embedding_split_axis
        else:
            self.split_axis = -1
        super(Embedding, self).__init__(vocab_size, embed_size, padding_idx=0)
        for param in self.parameters():
            nn.init.uniform_(param, a=-0.05, b=0.05)
            # W = np.load('/tank/model_zoo/dlrm_baseline_params_emb16/embedding_weight.npy')
            # param.data = flow.tensor(W, requires_grad=True)

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
    def __init__(self, vocab_size, embed_size, args):
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


embd_dict = {
    'OneEmbedding': OneEmbedding,
    'Embedding': Embedding,
}


class DLRMModule(nn.Module):
    def __init__(self, args):
        super(DLRMModule, self).__init__()
        self.bottom_mlp = MLP(args.num_dense_fields, args.bottom_mlp)
        self.embedding = embd_dict[args.embedding_type](args.vocab_size, args.embedding_vec_size, args)
        self.interaction = Interaction(args.interaction_type, args.interaction_itself, args.num_sparse_fields)
        feature_size = self.interaction.output_feature_size(args.embedding_vec_size, args.bottom_mlp[-1])        
        self.top_mlp = MLP(feature_size, args.top_mlp+[1], skip_final_activation=True)


    def forward(self, dense_fields, sparse_fields) -> flow.Tensor:
        dense_fields = flow.log(dense_fields + 1.0)
        dense_fields = self.bottom_mlp(dense_fields)
        sparse_fields = flow.cast(sparse_fields, flow.int64)
        embedding = self.embedding(sparse_fields)
        embedding = embedding.view(-1, embedding.shape[-1] * embedding.shape[-2])
        features = self.interaction(dense_fields, embedding)
        scores = self.top_mlp(features)
        return scores


def make_dlrm_module(args):
    model = DLRMModule(args)
    return model
