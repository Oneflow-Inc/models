import oneflow.experimental as flow
import oneflow.experimental.nn as nn
import oneflow.experimental.nn.functional as F

def init_method_const(tensor):
    return nn.init.constant_(tensor, 0.0)

def init_method_normal(tensor, sigma):
    """Init method based on N(0, sigma)."""
    return nn.init.normal_(tensor, mean=0.0, std=sigma)

def scaled_init_method_normal(tensor, sigma, num_layers):
    """Init method based on N(0, sigma/sqrt(2*num_layers)."""
    std = sigma / math.sqrt(2.0 * num_layers)
    return nn.init.normal_(tensor, mean=0.0, std=std)


class ColLinear(nn.Module):
    def __init__(
        input_size,
        output_size,
        need_gelu=False,
        bias_gelu_fusion=True,
    ):
        args = get_args()
        # TODO(): 这里之前的脚本中get_linear_params中传递了x.dtype，可能有问题
        # TODO(dis, done)
        # weight_parallel_dist=distribute.get_col_linear_weight_parallel_dist(),
        self.w = nn.Parameter(flow.Tensor((input_size, output_size)))
        init_method_normal(self.w, sigma=args.init_method_std)
        # TODO(dis, done)
        # bias_parallel_dist=distribute.get_col_linear_bias_parallel_dist(),
        self.b = nn.Parameter(flow.Tensor((output_size,)))
        init_method_const(self.b)
        self.need_gelu = need_gelu
        self.bias_gelu_fusion = bias_gelu_fusion
    
    def forward(self, x):
        # 2d sbp sig: [S(0), B] x [B, S(1)] -> [S(0), S(1)]
        # data grad 2d sbp sig: [S(0), S(1)] x [B, S(0)](transposed) -> [S(0), P] -> [S(0), B]
        # TODO(dis): 2d sbp cast, p2b, 显示控制AllReduce时机的
        # x = distribute.backward_p2b_parallel_cast(x)
        # TODO(dis): grap_sbp 写法2
        # x.to(grad_sbp=["S(0)", "B"])
        x = flow.matmul(x, w)
        if self.need_gelu:
            if self.bias_gelu_fusion:
                # TODO: megatron使用的torch.jit.script，参见Megatron-LM\megatron\model\fused_bias_gelu.py
                # 后面把这个方法迁移到functional
                x = F.fused_bias_add_gelu(x, self.b, data_format="NHC")
            else:
                x += self.b
                x = flow.gelu(x)
        else:
            x += self.b
    
        return x


class RowLinear(nn.Module):
    def __init__(
        input_size,
        output_size,
        dropout_rate=0.1,
        bias_dropout_fusion=True,
    ):
        # TODO(): 这里之前的脚本中get_linear_params中传递了x.dtype，可能有问题
        # TODO(dis, done)
        # weight_parallel_dist=distribute.get_row_linear_weight_parallel_dist(),
        self.w = nn.Parameter(flow.Tensor((input_size, output_size)))
        scaled_init_method_normal(self.w, sigma=args.init_method_std)
        # TODO(dis, done)
        # bias_parallel_dist=distribute.get_row_linear_bias_parallel_dist(),
        self.b = nn.Parameter(flow.Tensor((output_size,)))
        init_method_const(self.b)
        self.bias_dropout_fusion = bias_dropout_fusion
        self.dropout = flow.nn.Dropout(dropout_rate)

    def forward(self, x):
        # 2d sbp sig: [S(0), S(1)] x [B, S(0)] -> [S(0), P] -> [S(0), B]
        # data grad 2d sbp sig: [S(0), B] x [B, S(1)](transposed) -> [S(0), S(1)]
        x = flow.matmul(x, w)
        # TODO(dis)
        # x = distribute.forward_p2b_parallel_cast(x)
        if self.bias_dropout_fusion:
            # TODO: flow.nn.fused_bias_add_dropout是特殊实现的op，后面迁移到F.fused_bias_add_dropout
            x = F.fused_bias_add_dropout(x, b, data_format="NHC", rate=dropout_rate)
        else:
            x += self.b
            x = self.dropout(x)
    
        return x

class TransformerLayer(nn.Module):
    def __init__(
        self,
        layer_id,
        batch_size,
        seq_length,
        hidden_size,
    ):
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
        )
        self.mlp = MLP(
            batch_size,
            seq_length,
            hidden_size,
            args.hidden_dropout,
        )

        # TODO(dis, done)
        self.norm1 = flow.nn.LayerNorm(normalized_shape=(hidden_size,), eps=1e-5, elementwise_affine=True)
        self.norm2 = flow.nn.LayerNorm(normalized_shape=(hidden_size,), eps=1e-5, elementwise_affine=True)

    def forward(self, hidden_states):
        """
        hidden_states shape: (batch_size, seq_length, hidden_size)
        data parallel sbp: S(0)
        2d sbp: [S(0), B]
        """
        assert len(hidden_states.shape) == 3
        assert hidden_states.shape[-1] == self.hidden_size
        assert np.prod(hidden_states.shape[:-1]) == self.batch_size * self.seq_length

        h = hidden_states
        # TODO(dis)
        # 避免h被checkpointing释放掉
        # h = flow.identity(h)

    # TODO(dis, done)
    # with flow.experimental.scope.config(
    #     checkpointing=self.checkpoint_activations
    # ):
        # attention
        h = h + self.attn(self.norm1(h))
        # mlp
        h = h + self.mlp(self.norm2(h))

        return h

class SelfAttention(nn.Module):
    def __init__(
        self,
        layer_id,
        batch_size,
        seq_length,
        hidden_size,
        hidden_dropout_rate,
    ):
        self.layer_id = layer_id
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.hidden_size = hidden_size
        self.hidden_dropout_rate = hidden_dropout_rate

        args = get_args()
        self.num_heads = args.num_attention_heads
        self.head_size = args.hidden_size // args.num_attention_heads
        self.attention_dropout_rate = args.attention_dropout
        self.scale_tril_softmax_dropout_fusion = args.scale_tril_softmax_dropout_fusion
        self.bias_dropout_fusion = args.bias_dropout_fusion
        self.multihead_attention_fusion = args.multihead_attention_fusion

        self.norm_factor = math.sqrt(float(self.head_size))
        self.coeff = 1.0
        if args.apply_query_key_layer_scaling:
            self.coeff = float(self.layer_id)
            self.norm_factor *= self.coeff
        
        self.col_linear = ColLinear(self.hidden_size, self.hidden_size*3)

        self.dropout = flow.nn.Dropout(p=self.attention_dropout_rate)

        self.row_linear = RowLinear(
            self.hidden_size,
            self.hidden_size,
            dropout_rate=self.hidden_dropout_rate,
            bias_dropout_fusion=self.bias_dropout_fusion
            )

    def _query_key_value(self, h):
        """
        Split input to q, k, v and split hidden states into heads,
            shape: (batch_size, seq_length, hidden_size)
                -> (batch_size, seq_length, num_attn_heads, head_size)
                -> (batch_size, num_attn_heads, seq_length, head_size)
        """
        assert len(h.shape) == 3

        # Note: 3 is between num_heads and head_size that ensure the features of heads of q, k, v is contiguously arranged
        new_shape = (
            h.shape[0],
            h.shape[1],
            self.num_heads,
            3 * self.head_size,
        )
        if h.shape[0] == self.seq_length and h.shape[1] == self.batch_size:
            perm = [1, 2, 0, 3]
        elif h.shape[0] == self.batch_size and h.shape[1] == self.seq_length:
            perm = [0, 2, 1, 3]
        else:
            raise ValueError

        h = flow.reshape(h, new_shape)
        q, k, v = (
            h[:, :, :, i * self.head_size:(i + 1) * self.head_size].permute(*perm),
            for i in range(3)
        )
        return q, k, v

    def _multihead_attn(self, q, k, v):
        """
        q, k, v shape: (batch_size, num_attn_heads, seq_length, head_size)
        """
        assert all(len(x.shape) == 4 for x in (q, k, v))
        assert all(x.shape[0] == self.batch_size for x in (q, k, v))
        assert all(x.shape[1] == self.num_heads for x in (q, k, v))
        assert all(x.shape[2] == self.seq_length for x in (q, k, v))
        assert all(x.shape[3] == self.head_size for x in (q, k, v))

        # q * k: batch_matmul
        # shape sig: (b, n, s, h) x (b, n, h, s)(transposed) -> (b, n, s, s)
        # data parallel sbp sig: S(0) x S(0) -> S(0)
        # 2d sbp sig: [S(0), S(1)] x [S(0), S(1)] -> [S(0), S(1)]
        alpha = 1.0 / self.norm_factor
        qmk = flow.matmul(q, k.transpose(-2, -1)) * alpha

        qmk = self._tril_softmax_dropout(qmk)
        # w * v: batch_matmul
        # shape sig: (b, n, s, s) x (b, n, s, h) -> (b, n, s, h)
        # data parallel sbp sig: S(0) x S(0) -> S(0)
        # 2d sbp sig: [S(0), S(1)] x [S(0), S(1)] -> [S(0), S(1)]
        return flow.matmul(qmk, v)

    def _tril_softmax_dropout(self, x):
        if self.scale_tril_softmax_dropout_fusion:
            # TODO：特殊的算子fuse优化，flow.math.fused_scale_tril_softmax_dropout，
            # 后面迁移到F.fused_scale_tril_softmax_dropout
            x = F.fused_scale_tril_softmax_dropout(
                x,
                diagonal=0,
                scale=self.coeff,
                fill_value=float("-inf"),
                rate=self.attention_dropout_rate,
            )
        else:
            # TODO：特殊的算子fuse优化，flow.math.fused_scale_tril，
            # 后面迁移到F.fused_scale_tril
            x = F.fused_scale_tril(
                x, fill_value=float("-inf"), scale=self.coeff,
            )
            x = flow.softmax(x, dim=-1)
            x = self.dropout(x)

        return x

    def _fused_multihead_attn(self, h):
        assert len(h.shape) == 3
        assert h.shape[0] == self.seq_length
        assert h.shape[1] == self.batch_size
        assert h.shape[2] == self.hidden_size * 3

        qmk, v = flow.nn.fused_self_attention_query_mul_key_and_value(
            h, head_size=self.head_size, alpha=(1.0 / self.norm_factor)
        )
        qmk = self.tril_softmax_dropout(qmk)
        return flow.matmul(qmk, v)

    def forward(self, hidden_states):
        # hidden_states shape: (batch_size, seq_length, hidden_size)
        # or (seq_length, batch_size, hidden_size) [seq_len dim leading]
        # data parallel sbp: S(0)
        # 2d sbp: [S(0), B]
        assert len(hidden_states.shape) == 3
        assert hidden_states.shape[-1] == self.hidden_size
        if (
            hidden_states.shape[0] == self.batch_size
            and hidden_states.shape[1] == self.seq_length
        ):
            is_seq_len_dim_leading = False
        elif (
            hidden_states.shape[0] == self.seq_length
            and hidden_states.shape[1] == self.batch_size
        ):
            is_seq_len_dim_leading = True
        else:
            raise ValueError(f"invalid hidden states shape {hidden_states.shape}")

        h = hidden_states
        h = self.col_linear(h)
        if self.multihead_attention_fusion:
            # TODO(): 这是利用stride实现的优化，当前没有stride，所以自己创建了一个特殊的op来做
            # 待后面支持stride后，再支持这个优化
            raise NotImplementedError
            h = self._fused_multihead_attn(h)
        else:
            q, k, v = self._query_key_value(h)
            h = self._multihead_attn(q, k, v)

        if is_seq_len_dim_leading:
            # (b, n, s, h) -> (s, b, n, h)
            h = h.permute(2, 0, 1, 3)
        else:
            # (b, n, s, h) -> (b, s, n, h)
            h = h.permute(0, 2, 1, 3)

        # (b, s, n, h) -> (b, s, H) or (s, b, n, h) -> (s, b, H)
        h = h.flatten(start_dim=2)
        h = self.row_linear(h)

        return h

class MLP(nn.Module):
    def __init__(
        self,
        batch_size,
        seq_length,
        hidden_size,
        hidden_dropout_rate,
    ):
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.hidden_size = hidden_size
        self.hidden_dropout_rate = hidden_dropout_rate

        args = get_args()
        self.bias_gelu_fusion = args.bias_gelu_fusion
        self.bias_dropout_fusion = args.bias_dropout_fusion

        self.col_linear = ColLinear(self.hidden_size, self.hidden_size*4,
            need_gelu=True, bias_gelu_fusion=self.bias_gelu_fusion)

        self.row_linear = RowLinear(
            self.hidden_size,
            self.hidden_size,
            dropout_rate=self.hidden_dropout_rate,
            bias_dropout_fusion=self.bias_dropout_fusion
            )

    def forward(self, hidden_states):
        # hidden_states shape: (batch_size, seq_length, hidden_size)
        # data parallel sbp: S(0)
        # 2d sbp: [S(0), B]
        assert hidden_states.shape[-1] == self.hidden_size

        h = hidden_states
        h = self.col_linear(h)
        h = self.row_linear(h)

        # output hidden states shape: (batch_size * seq_length, hidden_size)
        # data parallel sbp: S(0)
        # 2d sbp: [S(0), B]
        return h
