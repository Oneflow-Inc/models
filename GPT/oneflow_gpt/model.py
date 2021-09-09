import math
import oneflow as flow

from oneflow_gpt import distribute as dist
from oneflow_gpt.config import get_args


class GPTModel(flow.nn.Module):
    def __init__(self):
        super().__init__()
        args = get_args()
        self.batch_size = args.global_batch_size // args.num_accumulation_steps
        self.seq_length = args.seq_length
        self.hidden_size = args.hidden_size

        if args.fp16:
            dtype = flow.float16
        else:
            dtype = flow.float32

        self.embedding = Embedding(
            self.seq_length, self.hidden_size, args.padded_vocab_size, dtype
        )
        self.transformer = Transformer(self.hidden_size, dtype)
        self.logits = Logits()

    def forward(self, tokens):
        # tokens shape: (batch_size, seq_length)
        # sbp: [S(0), B]
        assert tokens.ndim == 2
        assert tokens.shape[0] == self.batch_size
        assert tokens.shape[1] == self.seq_length

        hidden_states = self.embedding(tokens)
        h = self.transformer(hidden_states)

        assert h.shape[0] == self.batch_size
        assert h.shape[1] == self.seq_length
        assert h.shape[2] == self.hidden_size
        return self.logits(h, self.embedding.wte)


class Logits(flow.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, hidden_states, word_embeddings):
        assert hidden_states.ndim == 3

        w = word_embeddings.to_consistent(placement=hidden_states.placement)
        # h.grad.sbp: [S(0), P] -> [S(0), B]
        h = hidden_states.to_consistent(grad_sbp=hidden_states.sbp)
        # shape sign: (B * S, H) x (H, V) -> (B * S, V)
        # matmul fwd sbp sign: [S(0), B] x [B, S(1)] (wte.T) -> [S(0), S(1)]
        # bwd h.grad sbp sign: [S(0), S(1)] (lgs.grad) x [B, S(0)] (wte) -> [S(0), P] (h.grad)
        lgs = flow._C.matmul(h, w, transpose_b=True)
        return lgs


class Embedding(flow.nn.Module):
    def __init__(self, seq_length, hidden_size, vocab_size, dtype):
        super().__init__()
        self.seq_length = seq_length
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size

        args = get_args()
        self.dropout = flow.nn.Dropout(p=args.hidden_dropout)

        # word token embedding shape (vocab_size, hidden_size)
        # sbp: [B, S(0)]
        self.wte = flow.nn.Parameter(
            flow.empty(
                (self.vocab_size, self.hidden_size),
                dtype=dtype,
                placement=dist.get_layer_placement(0),
                sbp=dist.get_nd_sbp([flow.sbp.broadcast, flow.sbp.split(0)]),
            )
        )

        # word position embedding shape (seq_len, hidden_size)
        # sbp: [B, B]
        self.wpe = flow.nn.Parameter(
            flow.empty(
                (self.seq_length, self.hidden_size),
                dtype=dtype,
                placement=dist.get_layer_placement(0),
                sbp=dist.get_nd_sbp([flow.sbp.broadcast, flow.sbp.broadcast]),
            )
        )

        flow.nn.init.normal_(self.wte)
        flow.nn.init.normal_(self.wpe)

    def forward(self, tokens):
        # tokens shape: (batch_size, seq_len)
        # sbp: [S(0), B]
        assert tokens.ndim == 2
        assert tokens.shape[-1] == self.seq_length

        # wte.grad: [P, S(0)]  -> [B, S(0)]
        wte = self.wte.to_consistent(grad_sbp=self.wte.sbp)
        # gather forward sbp sign: [B, S(0)] x [S(0), B] -> [S(0), P]
        # backward sbp sign:
        # [S(0), B] (h.grad) x [S(0), B] (tokens) x [B, S(0)] (wte) -> [P, S(0)] (wte.grad)
        h = flow._C.gather(wte, tokens, axis=0)
        # hidden_states shape: (batch_size, sel_len, hidden_size)
        # hidden_states: [S(0), P] -> [S(0), B]
        h = h.to_consistent(sbp=dist.get_hidden_sbp())
        # (h + self.wpe) will apply broadcast_add,
        # shape sign: (batch_size, sel_len, hidden_size) + (sel_len, hidden_size)
        #         -> (batch_size, sel_len, hidden_size)
        # sbp sign: [S(0), B] + [B, B] -> [S(0), B]
        return self.dropout(h + self.wpe)


def init_method_normal(sigma):
    """Init method based on N(0, sigma)."""

    def init_(tensor):
        return flow.nn.init.normal_(tensor, mean=0.0, std=sigma)

    return init_


def scaled_init_method_normal(sigma, num_layers):
    """Init method based on N(0, sigma/sqrt(2*num_layers)."""
    std = sigma / math.sqrt(2.0 * num_layers)

    def init_(tensor):
        return flow.nn.init.normal_(tensor, mean=0.0, std=std)

    return init_


class Transformer(flow.nn.Module):
    def __init__(self, hidden_size, dtype):
        super().__init__()
        self.hidden_size = hidden_size
        self.dtype = dtype

        args = get_args()
        self.is_seq_len_dim_leading = True if args.multihead_attention_fusion else False
        self.num_layers = args.num_layers

        self._build_layers(args.init_method_std)
        self.layernorm_f = LayerNorm(-1, (self.hidden_size,), dtype)

    def _build_layers_v2(self, init_method_std):
        self.layers = flow.nn.ModuleList(
            [
                TransformerLayer(
                    i,
                    self.hidden_size,
                    self.is_seq_len_dim_leading,
                    self.dtype,
                    init_method=init_method_normal(init_method_std),
                    output_layer_init_method=scaled_init_method_normal(
                        init_method_std, self.num_layers
                    ),
                )
                for i in range(self.num_layers)
            ]
        )

    def _get_layer_v2(self, layer_idx):
        return self.layers[layer_idx]

    def _build_layers(self, init_method_std):
        for i in range(self.num_layers):
            setattr(
                self,
                f"layer_{i}",
                TransformerLayer(
                    i,
                    self.hidden_size,
                    self.is_seq_len_dim_leading,
                    self.dtype,
                    init_method=init_method_normal(init_method_std),
                    output_layer_init_method=scaled_init_method_normal(
                        init_method_std, self.num_layers
                    ),
                ),
            )
            setattr(self, f"layer_checkpoint_{i}", ActivationCheckpointing(i))

    def _get_layer(self, layer_idx):
        layer = getattr(self, f"layer_{layer_idx}")
        checkpoint = getattr(self, f"layer_checkpoint_{layer_idx}")
        return layer, checkpoint

    def forward(self, hidden_states):
        # hidden_states shape: (batch_size, seq_length, hidden_size)
        # sbp: [S(0), B]
        assert hidden_states.ndim == 3
        assert hidden_states.shape[-1] == self.hidden_size

        if self.is_seq_len_dim_leading:
            h = hidden_states.transpose(0, 1)
        else:
            h = hidden_states

        for i in range(self.num_layers):
            layer, checkpoint = self._get_layer(i)
            h = layer(checkpoint(h))

        h = self.layernorm_f(h)

        assert h.ndim == 3
        if self.is_seq_len_dim_leading:
            h = h.transpose(0, 1)

        return h


class ActivationCheckpointing(flow.nn.Module):
    def __init__(self, layer_idx):
        super().__init__()
        self.layer_idx = layer_idx

    def forward(self, x):
        x = x.to_consistent(placement=dist.get_layer_placement(self.layer_idx))
        return flow._C.identity(x)


class TransformerLayer(flow.nn.Module):
    def __init__(
        self,
        layer_idx,
        hidden_size,
        is_seq_len_dim_leading,
        dtype,
        init_method,
        output_layer_init_method,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.layer_idx = layer_idx

        args = get_args()
        self.attn = SelfAttention(
            layer_idx,
            hidden_size,
            is_seq_len_dim_leading,
            dtype,
            args.hidden_dropout,
            init_method,
            output_layer_init_method,
        )
        self.mlp = MLP(
            layer_idx,
            hidden_size,
            dtype,
            args.hidden_dropout,
            init_method,
            output_layer_init_method,
        )

        self.layernorm_1 = LayerNorm(layer_idx, (self.hidden_size,), dtype)
        self.layernorm_2 = LayerNorm(layer_idx, (self.hidden_size,), dtype)

    def forward(self, hidden_states):
        # hidden_states shape: (batch_size, seq_length, hidden_size)
        # sbp: [S(0), B]
        assert hidden_states.ndim == 3
        assert hidden_states.shape[-1] == self.hidden_size
        h = hidden_states

        norm1 = self.layernorm_1(h)
        h = h + self.attn(norm1)

        norm2 = self.layernorm_2(h)
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
            self.coeff = float(layer_idx + 1)
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
            flow._C.transpose(
                h[:, :, :, (i * self.head_size) : ((i + 1) * self.head_size)],
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
        qmk = flow._C.matmul(q, k, transpose_b=True, alpha=(1.0 / self.norm_factor))
        qmk = self.tril_softmax_dropout(qmk)
        # w * v: batch_matmul
        # shape sign: (b, n, s, s) x (b, n, s, h) -> (b, n, s, h)
        # sbp sign: [S(0), S(1)] x [S(0), S(1)] -> [S(0), S(1)]
        return flow._C.matmul(qmk, v)

    def tril_softmax_dropout(self, x):
        if self.scale_tril_softmax_dropout_fusion:
            x = flow._C.fused_scale_tril_softmax_dropout(
                x,
                diagonal=0,
                scale=self.coeff,
                fill_value=float("-inf"),
                rate=self.attention_dropout_rate,
            )
        else:
            x = flow._C.fused_scale_tril(x, fill_value=float("-inf"), scale=self.coeff,)
            x = flow._C.softmax(x)
            x = self.multihead_attn_dropout(x)

        return x

    def fused_multihead_attn(self, h):
        qmk, v = flow._C.fused_self_attention_query_mul_key_and_value(
            h, head_size=self.head_size, alpha=(1.0 / self.norm_factor)
        )
        qmk = self.tril_softmax_dropout(qmk)
        return flow._C.matmul(qmk, v)

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
            h = flow._C.transpose(h, perm=(2, 0, 1, 3))
        else:
            # (batch_size, num_heads, seq_len, head_size) -> (batch_size, seq_len, num_heads, head_size)
            h = flow._C.transpose(h, perm=(0, 2, 1, 3))

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
        super().__init__()
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
            self.hidden_size * 4,
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
            flow.empty(
                normalized_shape,
                dtype=dtype,
                placement=dist.get_layer_placement(layer_idx),
                sbp=dist.get_nd_sbp([flow.sbp.broadcast, flow.sbp.broadcast]),
            )
        )
        flow.nn.init.zeros_(self.beta)

        self.gamma = flow.nn.Parameter(
            flow.empty(
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
        y = flow._C.layer_norm_affine(
            x,
            self.gamma,
            self.beta,
            begin_norm_axis=begin_norm_axis,
            begin_params_axis=begin_params_axis,
            epsilon=self.epsilon,
        )
        return y


class ColumnParallelLinear(flow.nn.Module):
    def __init__(
        self, layer_idx, input_size, output_size, dtype, init_method, need_gelu=False
    ):
        super().__init__()
        self.need_gelu = need_gelu

        args = get_args()
        self.bias_gelu_fusion = args.bias_gelu_fusion

        # col parallel linear weight sbp: [B, S(1)]
        self.weight = flow.nn.Parameter(
            flow.empty(
                (input_size, output_size),
                dtype=dtype,
                placement=dist.get_layer_placement(layer_idx),
                sbp=dist.get_nd_sbp([flow.sbp.broadcast, flow.sbp.split(1)]),
            )
        )
        init_method(self.weight)

        # col parallel linear bias sbp: [B, S(0)]
        self.bias = flow.nn.Parameter(
            flow.empty(
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
        x = flow._C.matmul(x, self.weight)
        if self.need_gelu:
            if self.bias_gelu_fusion:
                x = flow._C.fused_bias_add_gelu(x, self.bias, axis=x.ndim - 1)
            else:
                x = x + self.bias
                x = flow._C.gelu(x)
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
            flow.empty(
                (input_size, output_size),
                dtype=dtype,
                placement=dist.get_layer_placement(layer_idx),
                sbp=dist.get_nd_sbp([flow.sbp.broadcast, flow.sbp.split(0)]),
            )
        )
        init_method(self.weight)

        # col parallel linear bias sbp: [B, B]
        self.bias = flow.nn.Parameter(
            flow.empty(
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
        x = flow._C.matmul(x, self.weight)
        # x.sbp: [S(0), P] -> [S(0), B]
        x = x.to_consistent(sbp=dist.get_hidden_sbp())
        if self.bias_dropout_fusion:
            x = flow._C.fused_bias_add_dropout(
                x, self.bias, p=self.dropout_rate, axis=x.ndim - 1
            )
        else:
            x = x + self.bias
            x = self.dropout(x)

        return x


class ParallelSparseSoftmaxCrossEntropyLoss(flow.nn.Module):
    def __init__(self):
        super().__init__()

    def __call__(self, logits, labels):
        # logits shape: (batch_size, seq_length, vocab_size)
        # sbp: [S(0), S(2)]
        # labels shape: (batch_size, seq_length)
        # sbp: [S(0), B]
        assert logits.ndim == 3
        assert labels.ndim == 2
        assert logits.shape[0:2] == labels.shape

        # loss = flow._C.sparse_softmax_cross_entropy_with_logits(labels, logits)
        loss = flow._C.sparse_softmax_cross_entropy(
            logits, labels, depth=logits.shape[-1]
        )

        # if not logits.has_s1_sbp and logits.is_half_dtype:
        #     loss = flow.amp_white_identity(loss)

        return loss.mean()
