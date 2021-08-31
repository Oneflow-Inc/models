import pickle
import numpy as np
import argparse
import math


def _parse_args():
    parser = argparse.ArgumentParser("flags for eval lazy and eager test bert")
    parser.add_argument(
        "--mode",
        type=str,
        help="lazy or eager: run lazy mode first and then run eager mode",
    )
    return parser.parse_args()


def run_lazy():

    import oneflow.compatible.single_client as flow
    import oneflow.core.operator.op_conf_pb2 as op_conf_util

    def CreateInitializer(std):
        return flow.truncated_normal(std)

    def _CreateAddrFromAttentionMask(
        attention_mask_blob, from_seq_length, to_seq_length
    ):
        attention_mask_blob = flow.reshape(
            attention_mask_blob, [-1, 1, from_seq_length, to_seq_length]
        )
        attention_mask_blob = flow.cast(attention_mask_blob, dtype=flow.float)
        addr_blob = (attention_mask_blob - 1.0) * 10000.0
        return addr_blob

    def lazy_AttentionLayer(
        from_blob,
        to_blob,
        mask_blob,
        num_attention_heads=1,
        size_per_head=512,
        query_act=op_conf_util.kNone,
        key_act=op_conf_util.kNone,
        value_act=op_conf_util.kNone,
        attention_probs_dropout_prob=0.0,
        initializer_range=0.02,
        do_return_2d_tensor=False,
        batch_size=None,
        from_seq_length=None,
        to_seq_length=None,
    ):
        def TransposeForScores(input_blob, num_attention_heads, seq_length, width):
            output_blob = flow.reshape(
                input_blob, [-1, seq_length, num_attention_heads, width]
            )
            output_blob = flow.transpose(output_blob, perm=[0, 2, 1, 3])
            return output_blob

        from_blob_2d = flow.reshape(
            from_blob, [-1, num_attention_heads * size_per_head]
        )
        to_blob_2d = flow.reshape(to_blob, [-1, num_attention_heads * size_per_head])

        query_blob, query_w, query_b = _FullyConnected(
            from_blob_2d,
            input_size=num_attention_heads * size_per_head,
            units=num_attention_heads * size_per_head,
            activation=query_act,
            name="query",
            weight_initializer=CreateInitializer(initializer_range),
        )

        key_blob, key_w, key_b = _FullyConnected(
            to_blob_2d,
            input_size=num_attention_heads * size_per_head,
            units=num_attention_heads * size_per_head,
            activation=key_act,
            name="key",
            weight_initializer=CreateInitializer(initializer_range),
        )

        value_blob, value_w, value_b = _FullyConnected(
            to_blob_2d,
            input_size=num_attention_heads * size_per_head,
            units=num_attention_heads * size_per_head,
            activation=value_act,
            name="value",
            weight_initializer=CreateInitializer(initializer_range),
        )

        query_blob = TransposeForScores(
            query_blob, num_attention_heads, from_seq_length, size_per_head
        )
        key_blob = TransposeForScores(
            key_blob, num_attention_heads, to_seq_length, size_per_head
        )

        attention_scores_blob = flow.matmul(query_blob, key_blob, transpose_b=True)
        attention_scores_blob = attention_scores_blob * (
            1.0 / math.sqrt(float(size_per_head))
        )

        attention_scores_blob = attention_scores_blob + _CreateAddrFromAttentionMask(
            mask_blob, from_seq_length, to_seq_length
        )
        attention_probs_blob = flow.nn.softmax(attention_scores_blob)
        attention_probs_blob = _Dropout(
            attention_probs_blob, attention_probs_dropout_prob
        )

        value_blob = flow.reshape(
            value_blob, [-1, to_seq_length, num_attention_heads, size_per_head]
        )
        value_blob = flow.transpose(value_blob, perm=[0, 2, 1, 3])
        context_blob = flow.matmul(attention_probs_blob, value_blob)
        context_blob = flow.transpose(context_blob, perm=[0, 2, 1, 3])

        mid_res = [query_blob, key_blob, value_blob]

        if do_return_2d_tensor:
            context_blob = flow.reshape(
                context_blob, [-1, num_attention_heads * size_per_head]
            )
        else:
            context_blob = flow.reshape(
                context_blob, [-1, from_seq_length, num_attention_heads * size_per_head]
            )
        return context_blob, [query_w, query_b, key_w, key_b, value_w, value_b], mid_res

    def _FullyConnected(
        input_blob,
        input_size,
        units,
        activation=None,
        name=None,
        weight_initializer=None,
    ):
        weight_blob = flow.get_variable(
            name=name + "-weight",
            shape=[input_size, units],
            dtype=input_blob.dtype,
            initializer=weight_initializer,
        )
        bias_blob = flow.get_variable(
            name=name + "-bias",
            shape=[units],
            dtype=input_blob.dtype,
            initializer=flow.constant_initializer(0.0),
        )
        output_blob = flow.matmul(input_blob, weight_blob)
        output_blob = flow.nn.bias_add(output_blob, bias_blob)
        return output_blob, weight_blob, bias_blob

    def _Dropout(input_blob, dropout_prob):
        if dropout_prob == 0.0:
            return input_blob
        return flow.nn.dropout(input_blob, rate=dropout_prob)

    # lazy fully connected layer
    func_config = flow.FunctionConfig()
    func_config.default_data_type(flow.float)

    # Scalar dimensions referenced here:
    #   B = batch size (number of sequences)
    #   F = `from_tensor` sequence length
    #   T = `to_tensor` sequence length
    #   N = `num_attention_heads`
    #   H = `size_per_head`
    # set initial dim
    B, F, T, N, H = 4, 12, 12, 4, 16

    @flow.global_function("predict", func_config)
    def fc_job(
        from_blob: flow.typing.Numpy.Placeholder((B, F, H * N)),
        to_blob: flow.typing.Numpy.Placeholder((B, T, H * N)),
        mask_blob: flow.typing.Numpy.Placeholder((B, F, T)),
    ):
        return lazy_AttentionLayer(
            from_blob,
            to_blob,
            mask_blob,
            num_attention_heads=N,
            size_per_head=H,
            do_return_2d_tensor=True,
            from_seq_length=F,
            to_seq_length=T,
        )

    # lazy out
    from_blob = np.random.normal(size=(B, F, H * N))
    to_blob = from_blob.copy()
    mask_blob = np.random.randint(0, 2, size=(B, F, T))
    input = [from_blob, to_blob, mask_blob]

    lazy_res, params, mid_res = fc_job(from_blob, to_blob, mask_blob).get()
    lazy_res = lazy_res.numpy()
    params = [i.numpy() for i in params]
    mid_res = [i.numpy() for i in mid_res]

    pickle_file = open("rst_param_cache.pkl", "wb")
    pickle.dump([input, lazy_res, params, mid_res], pickle_file)
    pickle_file.close()
    print("lazy input saved")


# ===============eager=============
def run_eager():
    import oneflow as flow
    from oneflow import nn

    class MultiHeadedAttention(nn.Module):
        """
        Take in model size and number of heads.
        """

        def __init__(self, h, d_model, params, dropout=0.0):
            super().__init__()
            assert d_model % h == 0

            # We assume d_v always equals d_k
            self.d_k = d_model // h
            self.h = h

            [query_w, query_b, key_w, key_b, value_w, value_b] = params
            query_fc = nn.Linear(d_model, d_model)
            query_fc.weight = nn.Parameter(flow.tensor(query_w.transpose()))
            query_fc.bias = nn.Parameter(flow.tensor(query_b))

            key_fc = nn.Linear(d_model, d_model)
            key_fc.weight = nn.Parameter(flow.tensor(key_w.transpose()))
            key_fc.bias = nn.Parameter(flow.tensor(key_b))

            value_fc = nn.Linear(d_model, d_model)
            value_fc.weight = nn.Parameter(flow.tensor(value_w.transpose()))
            value_fc.bias = nn.Parameter(flow.tensor(value_b))

            self.linear_layers = nn.ModuleList([query_fc, key_fc, value_fc])
            # self.output_linear = nn.Linear(d_model, d_model)
            self.attention = Attention()

            self.dropout = nn.Dropout(p=dropout)

        def forward(self, query, key, value, mask, mid_res):
            batch_size = query.size(0)  # 16

            query, key, value = [
                l(x).reshape(batch_size, -1, self.h, self.d_k).permute(0, 2, 1, 3)
                for l, x in zip(self.linear_layers, (query, key, value))
            ]
            is_equal = np.allclose(query.numpy(), mid_res[0], rtol=1e-4, atol=1e-4)
            # 2) Apply attention on all the projected vectors in batch.

            x, attn = self.attention(query, key, value, mask, self.dropout)

            # 3) "Concat" using a view and apply a final linear.
            res = x.transpose(1, 2).reshape(batch_size, -1, self.h * self.d_k)
            # res = self.output_linear(res)
            return res

    class Attention(nn.Module):
        """
        Compute 'Scaled Dot Product Attention
        """

        def __init__(self):
            super().__init__()
            self.softmax = nn.Softmax(dim=-1)

        def forward(self, query, key, value, mask=None, dropout=None):
            x = flow.matmul(query, key.transpose(-2, -1))
            scores = x / math.sqrt(query.size(-1))

            if mask is not None:
                mask = mask.unsqueeze(1)
                mask = mask.repeat((1, query.size(1), 1, 1))
                mask = flow.eq(mask, 0)
                scores = scores.masked_fill(mask, -1e9)

            p_attn = self.softmax(scores)

            if dropout is not None:
                p_attn = dropout(p_attn)

            return flow.matmul(p_attn, value), p_attn

    # eager attention
    B, F, T, N, H = 4, 12, 12, 4, 16
    print("reading saved pkl")
    pickle_file = open("rst_param_cache.pkl", "rb")
    [input, lazy_res, params, mid_res] = pickle.load(pickle_file)
    from_blob, to_blob, mask_blob = input
    eager_attention = MultiHeadedAttention(N, N * H, params)
    eager_res = eager_attention(
        flow.tensor(from_blob, dtype=flow.float32),
        flow.tensor(to_blob, dtype=flow.float32),
        flow.tensor(to_blob, dtype=flow.float32),
        flow.tensor(mask_blob, dtype=flow.float32),
        mid_res,
    )
    eager_res = eager_res.reshape(-1, N * H).numpy()
    is_equal = np.allclose(lazy_res, eager_res, rtol=1e-4, atol=1e-4)
    print(f"eager and lazy is equal? {is_equal}")


if __name__ == "__main__":
    args = _parse_args()
    if args.mode == "lazy":
        run_lazy()
    elif args.mode == "eager":
        run_eager()
    else:
        raise NotImplementedError
