'''
gpt-2 model参考megatron-lm
'''
import copy
import math

import oneflow as flow
import oneflow.nn as nn
import oneflow.nn.functional as F

from gpt_module_util import *
from config import get_args

class Embedding(nn.Module):
    def __init__(self, batch_size, seq_length, hidden_size, vocab_size):
        super().__init__()

        self.batch_size = batch_size
        self.seq_length = seq_length
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size

        args = get_args()
        self.embedding_dropout_rate = args.hidden_dropout
        self.use_fp16 = args.fp16

        # TODO(dis, done)
        # with distribute.layer_placement_scope(0):
        # wpe = flow.get_variable(
        #     "wpe",
        #     shape=(self.seq_length, self.hidden_size),
        #     initializer=self.wpe_initializer,
        # TODO(dis, done) parallel_distribution=distribute.get_wpe_parallel_dist(),
        # )
        # word position embedding
        self.wpe = nn.Parameter(flow.Tensor(self.seq_length, self.hidden_size))
        init_method_normal(self.wpe, sigma=args.init_method_std)
        # word token embedding
        self.wte = nn.Parameter(flow.Tensor(self.vocab_size, self.hidden_size))
        init_method_normal(self.wte, sigma=args.init_method_std)
        self.dropout = flow.nn.Dropout(self.embedding_dropout_rate)

    def forward(self, tokens):
        """
        tokens shape: (batch_size, seq_length)
        dp sbp: S(0)
        2d sbp: [S(0), B]
        """
        assert len(tokens.shape) == 2
        assert tokens.shape[0] == self.batch_size
        assert tokens.shape[1] == self.seq_length

        # TODO(dis, done)
        # with distribute.layer_placement_scope(0):
        # 2d sbp sig: [B, S(0)] x [S(0), B] -> [S(0), P] -> [S(0), B]
        # grad 2d sbp sig: [S(0), B](dy) x [S(0), B](index) x [B, S(0)](x)
        #                   -> [P, S(0)](dx) -> [B, S(0)](wte_grad)
        if self.use_fp16:
            # TODO(dis, done): amp
            # 标记 + cast, h = flow.gather(flow.amp_white_identity(wte), tokens)
            h = flow.gather(self.wte, tokens)
            # TODO(dis, done): amp
            # wpe也需要转下, wpe = flow.amp_white_identity(wpe)
        else:
            h = flow.gather(self.wte, tokens)

        # TODO(dis): 2d sbp
        # h = distribute.forward_p2b_parallel_cast(h) + wpe
        h = h + self.wpe
        h = self.dropout(h)

        return h, self.wte


class Transformer(nn.Module):
    def __init__(self, batch_size, seq_length, hidden_size):
        super().__init__()

        self.batch_size = batch_size
        self.seq_length = seq_length
        self.hidden_size = hidden_size

        args = get_args()
        self.multihead_attention_fusion = args.multihead_attention_fusion
        self.num_layers = args.num_layers
        self.layers = []
        for i in range(self.num_layers):
            self.layers.append(
                TransformerLayer(
                    i + 1,
                    batch_size,
                    seq_length,
                    hidden_size,
                )
            )
        
        # TODO(dis, done): params_parallel_dist=distribute.get_layernorm_params_parallel_dist()
        # 里面的affine关联的两个参数，要做2d sbp，配置是(B, B)
        self.layer_norm = flow.nn.LayerNorm(normalized_shape=(self.hidden_size,), eps=1e-5, elementwise_affine=True)

        if self.multihead_attention_fusion:
            self.permute = self.Permute()
    
    class Permute(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, h):
            return h.permute(1, 0, 2)

    def forward(self, hidden_states):
        """
        hidden_states shape: (batch_size, seq_length, hidden_size)
        data parallel sbp: S(0)
        2d sbp: [S(0), B]
        """
        assert len(hidden_states.shape) == 3
        assert hidden_states.shape[0] == self.batch_size
        assert hidden_states.shape[1] == self.seq_length
        assert hidden_states.shape[2] == self.hidden_size

        # if self.multihead_attention_fusion:
        # TODO(dis, done)
        # with distribute.layer_placement_scope(0):
            # [b, s, H] -> [s, b, H] for multihead_attention_fusion
            # h = hidden_states.permute(1, 0 , 2)
            # h = self.permute(h)
        # else:
        h = hidden_states

        for i in range(self.num_layers):
        # TODO(dis, done)
        # with distribute.layer_placement_scope(i):
            h = self.layers[i](h)

        # final layernorm
        # TODO(dis, done)
        # with distribute.layer_placement_scope(-1):
        h = self.layer_norm(h)

        return h

class Logits(nn.Module):
    def __init__(self):
        super().__init__()
        args = get_args() # config args
        self.batch_size = args.global_batch_size // args.num_accumulation_steps
        self.seq_length = args.seq_length
        self.hidden_size = args.hidden_size
        self.vocab_size = args.padded_vocab_size

    def forward(self, hidden_states, token_embeddings):
        """
        shape sig: (batch_size * seq_length, hidden_size) x (hidden_size, vocab_size)(transposed)
            -> (batch_size * seq_length, vocab_size)
        dp sbp sig: S(0) x B -> S(0)
        2d sbp sig: [S(0), B] x [B, S(1)](transposed) -> [S(0), S(1)]
        """
        assert len(hidden_states.shape) == 3
        assert np.prod(hidden_states.shape[0:2]) == self.batch_size * self.seq_length
        assert hidden_states.shape[-1] == self.hidden_size
        assert len(token_embeddings.shape) == 2
        assert token_embeddings.shape[0] == self.vocab_size
        assert token_embeddings.shape[1] == self.hidden_size

        # TODO(dis, done): with distribute.layer_placement_scope(-1):
        if (
            hidden_states.shape[0] == self.seq_length
            and hidden_states.shape[1] == self.batch_size
        ):
            # [s, b, H] -> [b, s, H]
            # old: h = flow.transpose(hidden_states, [1, 0, 2])
            h = hidden_states.permute(1, 0 , 2)
        elif (
            hidden_states.shape[0] == self.batch_size
            and hidden_states.shape[1] == self.seq_length
        ):
            h = hidden_states
        else:
            raise ValueError(f"invalid hidden states shape {hidden_states.shape}")

        # [s, b, H] or [b, s, H] -> [b * s, H]
        h = flow.flatten(h, start_dim=0, end_dim=1)
        # 2d sbp sig: [S(0), B] x [B, S(1)](transposed) -> [S(0), S(1)]
        # grad 2d sbp sig: [S(0), S(1)] x [B, S(0)] -> [S(0), P] -> [S(0), B]
        # TODO(dis): h = distribute.backward_p2b_parallel_cast(h)
        # suppose F.linear will transpose token_embeddings
        lgs = F.linear(h, token_embeddings)

        return lgs


class GPTModel(nn.Module):
    def __init__(self):
        super().__init__()
        args = get_args() # config args
        self.batch_size = args.global_batch_size // args.num_accumulation_steps
        self.seq_length = args.seq_length
        self.hidden_size = args.hidden_size
        self.vocab_size = args.padded_vocab_size

        self.embedding = Embedding(
            self.batch_size, self.seq_length, self.hidden_size, self.vocab_size
        )
        self.transformer = Transformer(
            self.batch_size, self.seq_length, self.hidden_size
        )
        self.logits = Logits()

    def forward(self, tokens):
        """
        tokens shape: (batch_size, seq_length)
        dp sbp: S(0)
        2d sbp: [S(0), B]
        """
        assert len(tokens.shape) == 2
        assert tokens.shape[0] == self.batch_size
        assert tokens.shape[1] == self.seq_length

        hidden_states, token_embeddings = self.embedding(tokens)
        h = self.transformer(hidden_states)
        lgs = self.logits(h, token_embeddings)

        return lgs

class SparseSoftmaxCrossEntropyLoss(nn.Module):
    def __init__(self):
        super().__init__()
        args = get_args()
        self.batch_size = args.global_batch_size // args.num_accumulation_steps
        self.seq_length = args.seq_length
        self.vocab_size = args.padded_vocab_size
        self.loss_module = flow.nn.CrossEntropyLoss(reduction="mean")

    def forward(self, logits, labels):
        """
        logits shape: (batch_size * seq_length, vocab_size)
        logits dp sbp: S(0)
        logits 2d sbp: [S(0), S(1)]
        labels shape: (batch_size, seq_length)
        labels dp sbp: S(0)
        labels 2d sbp: [S(0), B]
        """
        assert len(logits.shape) == 3 or len(logits.shape) == 2
        if len(logits.shape) == 3:
            assert logits.shape[0] == self.batch_size
            assert logits.shape[1] == self.seq_length
            assert logits.shape[2] == self.vocab_size
        elif len(logits.shape) == 2:
            assert logits.shape[0] == self.batch_size * self.seq_length
            assert logits.shape[1] == self.vocab_size
        else:
            raise ValueError(f"invalid logits shape {logits.shape}")

        assert len(labels.shape) == 2
        assert labels.shape[0] == self.batch_size
        assert labels.shape[1] == self.seq_length

    # TODO(dis, done)
    # with distribute.layer_placement_scope(-1):
        if len(logits.shape) == 2:
            labels = labels.flatten()

        # TODO(dis):
        # 如果模型并行，使用该loss op，是一个专门实现的loss op
        # 模型并行时使用的
        # TODO: 需要迁移到Module
        # if distribute.get_dist_util().tensor_model_parallel_size > 1:
        #     loss = flow.nn.distributed_sparse_softmax_cross_entropy_with_logits(
        #         labels, logits
        #     )
        #     loss = flow.math.reduce_mean(loss)
        # else:
            # 这个是非模型并行的版本
        loss = self.loss_module(
            logits, labels 
        )
            # TODO(dis, done): amp
            #loss = flow.amp_white_identity(loss)

        return loss
