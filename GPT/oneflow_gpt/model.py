import math
import numpy as np
import oneflow as flow

from oneflow_gpt import distribute as dist
from oneflow_gpt.config import get_args


class GPTModel(flow.nn.Module):
    def __init__(self, name):
        self.name = name

        args = get_args()
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

    def forward(self, tokens):
        """
        tokens shape: (batch_size, seq_length), sbp: [S(0), B]
        """
        assert len(tokens.shape) == 2
        assert tokens.shape[0] == self.batch_size
        assert tokens.shape[1] == self.seq_length

        hidden_states, token_embeddings = self.embedding(tokens)
        h = self.transformer(hidden_states)
        lgs = self.logits(h, token_embeddings)

        return lgs

    def logits(self, hidden_states, token_embeddings):
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

        if (
            hidden_states.shape[0] == self.seq_length
            and hidden_states.shape[1] == self.batch_size
        ):
            # [s, b, H] -> [b, s, H]
            h = flow.transpose(hidden_states, [1, 0, 2])
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
        h = distribute.backward_p2b_parallel_cast(h)
        lgs = flow.matmul(h, token_embeddings, transpose_b=True)

        return lgs


class Embedding(flow.nn.Module):
    def __init__(self, batch_size, seq_length, hidden_size, vocab_size):
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size

        args = get_args()
        self.dropout = flow.nn.Dropout(p=args.hidden_dropout)

        # word token embedding shape (vocab_size, hidden_size)
        # sbp: [B, S(0)]
        self.wte = flow.nn.Parameter(
            flow.Tensor(
                (self.vocab_size, self.hidden_size),
                placement=dist.get_wte_placement(),
                sbp=dist.get_nd_sbp(),
            )
        )

        # word position embedding shape (seq_len, hidden_size)
        # sbp: [B, B]
        self.wpe = flow.nn.Parameter(
            flow.Tensor(
                (self.seq_length, self.hidden_size),
                placement=dist.get_wpe_placement(),
                sbp=dist.get_nd_sbp(["S(0)", "B"]),
            )
        )

        flow.nn.init.normal_(self.wte)
        flow.nn.init.normal_(self.wpe)

        self.use_fp16 = args.fp16

    def forward(self, tokens):
        # tokens shape: (batch_size, seq_len), sbp: [S(0), B]
        assert tokens.ndim == 2
        assert tokens.shape[0] == self.batch_size
        assert tokens.shape[1] == self.seq_length

        # [P, S(0)] (wte.grad) -> [B, (0)] (wte.grad)
        wte = self.wte.to_consistent(grad_sbp=self.wte.sbp)
        # gather forward sbp signature: [B, S(0)] x [S(0), B] -> [S(0), P]
        # backward sbp signature:
        # [S(0), B] (h.grad) x [S(0), B] (tokens) x [B, S(0)] (wte) -> [P, S(0)] (wte.grad)
        h = flow.F.gather(wte, tokens)
        # hidden_states shape: (batch_size, sel_len, hidden_size)
        # [S(0), P] (hidden_states) -> [S(0), B] (hidden_states)
        h = h.to_consistent(sbp=dist.get_activation_sbp())
        # (h + self.wpe) will apply broadcast_add,
        # shape: (batch_size, sel_len, hidden_size) + (sel_len, hidden_size)
        #         -> (batch_size, sel_len, hidden_size)
        # sbp: [S(0), B] + [B, B] -> [S(0), B]
        return self.dropout(h + self.wpe), self.wte


class Transformer(object):
    def __init__(self, batch_size, seq_length, hidden_size):
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
                    f"h{i}",
                    i + 1,
                    batch_size,
                    seq_length,
                    hidden_size,
                    initializer=flow.random_normal_initializer(
                        stddev=args.init_method_std
                    ),
                    output_layer_initializer=flow.random_normal_initializer(
                        stddev=(args.init_method_std / math.sqrt(2.0 * self.num_layers))
                    ),
                )
            )

    def forward(self, hidden_states):
        # hidden_states shape: (batch_size, seq_length, hidden_size)
        # sbp: [S(0), B]
        assert hidden_states.ndim == 3
        assert hidden_states.shape[0] == self.batch_size
        assert hidden_states.shape[1] == self.seq_length
        assert hidden_states.shape[2] == self.hidden_size

        h = hidden_states
        for i in range(self.num_layers):
            h = self.layers[i](h)

        h = layernorm("layernorm_f", h)

        return layernorm("layernorm_f", h)


class TransformerLayer(object):
    def __init__(
        self,
        name,
        layer_id,
        batch_size,
        seq_length,
        hidden_size,
        initializer=None,
        output_layer_initializer=None,
    ):
        self.name = name
        self.layer_id = layer_id
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.hidden_size = hidden_size

        args = get_args()
        self.attn = SelfAttention(
            layer_id,
            batch_size,
            seq_length,
            hidden_size,
            args.hidden_dropout,
            initializer,
            output_layer_initializer,
        )
        self.mlp = MLP(
            batch_size,
            seq_length,
            hidden_size,
            args.hidden_dropout,
            initializer,
            output_layer_initializer,
        )

        self.checkpoint_activations = args.checkpoint_activations

    def forward(self, hidden_states):
        # hidden_states shape: (batch_size, seq_length, hidden_size)
        # nd_sbp: [S(0), B]
        assert hidden_states.ndim == 3
        assert hidden_states.shape[-1] == self.hidden_size
        assert np.prod(hidden_states.shape[:-1]) == self.batch_size * self.seq_length

        h = hidden_states
        with flow.scope.namespace(self.name):

            h = flow.identity(h)
            with flow.experimental.scope.config(
                checkpointing=self.checkpoint_activations
            ):
                # input layernorm
                norm1 = layernorm("layernorm_1", h)
                # attention
                h = h + self.attn(norm1)
                # output layernorm
                norm2 = layernorm("layernorm_2", h)
                # mlp
                h = h + self.mlp(norm2)

        return h


class SelfAttention(flow.nn.Module):
    def __init__(
        self,
        layer_idx,
        hidden_size,
        is_seq_len_dim_leading,
        dtype,
        hidden_dropout_rate,
        init_method,
        output_layer_init_method,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.is_seq_len_dim_leading = is_seq_len_dim_leading

        args = get_args()
        self.num_heads = args.num_attention_heads
        self.head_size = args.hidden_size // args.num_attention_heads
        self.attention_dropout_rate = args.attention_dropout
        self.scale_tril_softmax_dropout_fusion = args.scale_tril_softmax_dropout_fusion
        self.multihead_attention_fusion = args.multihead_attention_fusion

        if not self.scale_tril_softmax_dropout_fusion:
            self.multihead_attn_dropout = flow.nn.Dropout(p=self.attention_dropout_rate)

        self.norm_factor = math.sqrt(float(self.head_size))
        self.coeff = 1.0
        if args.apply_query_key_layer_scaling:
            self.coeff = float(layer_idx)
            self.norm_factor *= self.coeff

        self.c_attn = ColumnParallelLinear(
            layer_idx,
            self.hidden_size,
            self.hidden_size * 3,
            dtype,
            init_method,
            need_gelu=True,
        )

        self.c_proj = RowParallelLinear(
            layer_idx,
            self.hidden_size,
            self.hidden_size,
            dtype,
            output_layer_init_method,
            dropout_rate=hidden_dropout_rate,
        )

    def query_key_value(self, h):
        """
        Split input to q, k, v and split hidden states into heads,
            shape: (batch_size, seq_length, hidden_size)
                -> (batch_size, seq_length, num_attn_heads, head_size)
                -> (batch_size, num_attn_heads, seq_length, head_size)
        """
        # Note: 3 is between num_heads and head_size
        # that ensure the features of heads of q, k, v is contiguously arranged
        new_shape = (
            h.shape[0],
            h.shape[1],
            self.num_heads,
            3 * self.head_size,
        )

        if self.is_seq_len_dim_leading:
            # (seq_len, batch_size, num_heads, head_size) -> (batch_size, num_heads, seq_len, head_size)
            perm = [1, 2, 0, 3]
        else:
            # (batch_size, seq_len, num_heads, head_size) -> (batch_size, num_heads, seq_len, head_size)
            perm = [0, 2, 1, 3]

        h = h.view(*new_shape)
        q, k, v = (
            flow.F.transpose(
                h[:, :, :, (i * self.head_size):((i + 1) * self.head_size)],
                perm=perm,
            )
            for i in range(3)
        )
        return q, k, v

    def multihead_attn(self, q, k, v):
        # q, k, v shape: (batch_size, num_heads, seq_length, head_size)
        # q * k: batch_matmul
        # shape sign: (b, n, s, h) x (b, n, h, s) (k.T) -> (b, n, s, s)
        # sbp sign: [S(0), S(1)] x [S(0), S(1)] -> [S(0), S(1)]
        qmk = flow.matmul(q, k, transpose_b=True, alpha=(1.0 / self.norm_factor))
        qmk = self.tril_softmax_dropout(qmk)
        # w * v: batch_matmul
        # shape sign: (b, n, s, s) x (b, n, s, h) -> (b, n, s, h)
        # sbp sign: [S(0), S(1)] x [S(0), S(1)] -> [S(0), S(1)]
        return flow.matmul(qmk, v)

    def tril_softmax_dropout(self, x):
        if self.scale_tril_softmax_dropout_fusion:
            x = flow.F.fused_scale_tril_softmax_dropout(
                x,
                diagonal=0,
                scale=self.coeff,
                fill_value=float("-inf"),
                rate=self.attention_dropout_rate,
            )
        else:
            x = flow.F.fused_scale_tril(
                x, fill_value=float("-inf"), scale=self.coeff,
            )
            x = flow.F.softmax(x)
            x = self.multihead_attn_dropout(x)

        return x

    def fused_multihead_attn(self, h):
        qmk, v = flow.F.fused_self_attention_query_mul_key_and_value(
            h, head_size=self.head_size, alpha=(1.0 / self.norm_factor)
        )
        qmk = self.tril_softmax_dropout(qmk)
        return flow.matmul(qmk, v)

    def forward(self, hidden_states):
        # hidden_states shape: (batch_size, seq_len, hidden_size)
        # or (seq_len, batch_size, hidden_size) [seq_len dim leading]
        # sbp: [S(0), B]
        assert hidden_states.shape[-1] == self.hidden_size

        h = self.c_attn(hidden_states)

        if self.multihead_attention_fusion and self.is_seq_len_dim_leading:
            h = self.fused_multihead_attn(h)
        else:
            q, k, v = self.query_key_value(h)
            h = self.multihead_attn(q, k, v)

        if self.is_seq_len_dim_leading:
            # (batch_size, num_heads, seq_len, head_size) -> (seq_len, batch_size, num_heads, head_size)
            h = flow.F.transpose(h, perm=(2, 0, 1, 3))
        else:
            # (batch_size, num_heads, seq_len, head_size) -> (batch_size, seq_len, num_heads, head_size)
            h = flow.F.transpose(h, perm=(0, 2, 1, 3))

        h = self.c_proj(h.flatten(2))
        return h


class MLP(flow.nn.Module):
    def __init__(
        self,
        layer_idx,
        hidden_size,
        dtype,
        hidden_dropout_rate,
        init_method,
        output_layer_init_method,
    ):
        self.hidden_size = hidden_size

        self.c_fc = ColumnParallelLinear(
            layer_idx,
            self.hidden_size,
            self.hidden_size * 4,
            dtype,
            init_method,
            need_gelu=True,
        )

        self.c_proj = RowParallelLinear(
            layer_idx,
            self.hidden_size,
            self.hidden_size,
            dtype,
            output_layer_init_method,
            dropout_rate=hidden_dropout_rate,
        )

    def forward(self, hidden_states):
        # hidden_states shape: (batch_size, seq_length, hidden_size)
        # sbp: [S(0), B]
        assert hidden_states.shape[-1] == self.hidden_size
        h = self.c_fc(hidden_states)
        h = self.c_proj(h)
        return h


class LayerNorm(flow.nn.Module):
    def __init__(
        self, layer_idx, normalized_shape, dtype, eps=1e-5,
    ):
        super().__init__()
        self.normalized_shape = normalized_shape
        self.epsilon = eps

        self.beta = flow.nn.Parameter(
            flow.Tensor(
                normalized_shape,
                dtype=dtype,
                placement=dist.get_layer_placement(layer_idx),
                sbp=dist.get_nd_sbp([flow.sbp.broadcast, flow.sbp.broadcast]),
            )
        )
        flow.nn.init.zeros_(self.beta)

        self.gamma = flow.nn.Parameter(
            flow.Tensor(
                normalized_shape,
                dtype=dtype,
                placement=dist.get_layer_placement(layer_idx),
                sbp=dist.get_nd_sbp([flow.sbp.broadcast, flow.sbp.broadcast]),
            )
        )
        flow.nn.init.ones_(self.gamma)

    def forward(self, x):
        assert x.shape[-len(self.normalized_shape) :] == self.normalized_shape
        begin_norm_axis = x.ndim - len(self.normalized_shape)
        begin_params_axis = x.ndim - len(self.normalized_shape)
        y = flow.F.layer_norm_affine(
            x,
            self.gamma,
            self.beta,
            begin_norm_axis=begin_norm_axis,
            begin_params_axis=begin_params_axis,
            epsilon=self.epsilon,
        )
        return y


class ColumnParallelLinear(flow.nn.module):
    def __init__(
        self, layer_idx, input_size, output_size, dtype, init_method, need_gelu=False
    ):
        super().__init__()
        self.need_gelu = need_gelu

        args = get_args()
        self.bias_gelu_fusion = args.bias_gelu_fusion

        # col parallel linear weight sbp: [B, S(1)]
        self.weight = flow.nn.Parameter(
            flow.Tensor(
                (input_size, output_size),
                dtype=dtype,
                placement=dist.get_layer_placement(layer_idx),
                sbp=dist.get_nd_sbp([flow.sbp.broadcast, flow.sbp.split(1)]),
            )
        )
        init_method(self.weight)

        # col parallel linear bias sbp: [B, S(0)]
        self.bias = flow.nn.Parameter(
            flow.Tensor(
                (output_size,),
                dtype=dtype,
                placement=dist.get_layer_placement(layer_idx),
                sbp=dist.get_nd_sbp([flow.sbp.broadcast, flow.sbp.split(0)]),
            )
        )
        flow.nn.init.zeros_(self.bias)

    def forward(self, x):
        # x sbp: [S(0), B]
        # x.grad sbp: [S(0), P] -> [S(0), B]
        x = x.to_consistent(grad_sbp=x.sbp)
        # matmul sbp sign: [S(0), B] x [B, S(1)] -> [S(0), S(1)]
        # x.grad sbp sign: [S(0), S(1)] x [B, S(0)] (weight.T) -> [S(0), P]
        x = flow.F.matmul(x, self.weight)
        if self.need_gelu:
            if self.bias_gelu_fusion:
                x = flow.F.fused_bias_add_gelu(x, self.bias)
            else:
                x = x + self.bias
                x = flow.F.gelu(x)
        else:
            # broadcast_add shape sign: (input_size, output_size) + (output_size, )
            #                                = (input_size, output_size)
            # bias_add sbp sign: [S(0), S(1)] + [B, S(0)] = [S(0), S(1)]
            x = x + self.bias

        return x


class RowParallelLinear(flow.nn.Module):
    def __init__(
        self, layer_idx, input_size, output_size, dtype, init_method, dropout_rate,
    ):
        super().__init__()
        self.dropout_rate = dropout_rate

        args = get_args()
        self.bias_dropout_fusion = args.bias_dropout_fusion
        if not self.bias_dropout_fusion:
            self.dropout = flow.nn.Dropout(p=dropout_rate)

        # col parallel linear weight sbp: [B, S(0)]
        self.weight = flow.nn.Parameter(
            flow.Tensor(
                (input_size, output_size),
                dtype=dtype,
                placement=dist.get_layer_placement(layer_idx),
                sbp=dist.get_nd_sbp([flow.sbp.broadcast, flow.sbp.split(0)]),
            )
        )
        init_method(self.weight)

        # col parallel linear bias sbp: [B, B]
        self.bias = flow.nn.Parameter(
            flow.Tensor(
                (output_size,),
                dtype=dtype,
                placement=dist.get_layer_placement(layer_idx),
                sbp=dist.get_nd_sbp([flow.sbp.broadcast, flow.sbp.broadcast]),
            )
        )
        flow.nn.init.zeros_(self.bias)

    def forward(self, x):
        # x.sbp: [S(0), S(1)]
        # matmul sbp sign: [S(0), S(1)] x [B, S(0)] -> [S(0), P]
        # backward x.grad sbp sign: [S(0), B] x [B, S(1)] (weight.T) -> [S(0), S(1)]
        x = flow.F.matmul(x, self.weight)
        # x.sbp: [S(0), P] -> [S(0), B]
        x = x.to_consistent(sbp=dist.get_hidden_sbp())
        if self.bias_dropout_fusion:
            x = flow.F.fused_bias_add_dropout(x, self.bias, rate=self.dropout_rate)
        else:
            x = x + self.bias
            x = self.dropout(x)

        return x


class ParallelSparseSoftmaxCrossEntropyLoss(object):
    def __init__(self, name="loss"):
        self.name = name

        args = get_args()
        self.batch_size = args.global_batch_size // args.num_accumulation_steps
        self.seq_length = args.seq_length
        self.vocab_size = args.padded_vocab_size

    def __call__(self, logits, labels):
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

        with flow.scope.namespace(self.name):
            with distribute.layer_placement_scope(-1):
                if len(logits.shape) == 2:
                    labels = flow.flatten(labels)

                if distribute.get_dist_util().tensor_model_parallel_size > 1:
                    loss = flow.nn.distributed_sparse_softmax_cross_entropy_with_logits(
                        labels, logits
                    )
                else:
                    loss = flow.nn.sparse_softmax_cross_entropy_with_logits(
                        labels, logits
                    )
                    loss = flow.amp_white_identity(loss)

                loss = flow.math.reduce_mean(loss)

        return loss
