import pickle
import numpy as np
import argparse
import math


def _parse_args():
    parser = argparse.ArgumentParser("flags for eval lazy and eager test bert")
    parser.add_argument(
        "--mode",
        type=str,
        default="eager",
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
            name=name + '-weight',
            shape=[input_size, units],
            dtype=input_blob.dtype,
            initializer=weight_initializer,
        )
        bias_blob = flow.get_variable(
            name=name + '-bias',
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

    class BertSelfAttention(nn.Module):
        def __init__(
            self,
            num_attention_heads,
            hidden_size,
            seq_len,
            params,
            hidden_dropout_prob=0.0,
        ):
            super().__init__()

            self.num_attention_heads = num_attention_heads
            self.attention_head_size = int(hidden_size / num_attention_heads)
            self.all_head_size = num_attention_heads * self.attention_head_size
            self.seq_len = seq_len

            [query_w, query_b, key_w, key_b, value_w, value_b] = params
            self.query = nn.Linear(hidden_size, self.all_head_size)
            self.query.weight.data.copy_(flow.tensor(query_w.transpose()))
            self.query.bias.data.copy_(flow.tensor(query_b))

            self.key = nn.Linear(hidden_size, self.all_head_size)
            self.key.weight.data.copy_(flow.tensor(key_w.transpose()))
            self.key.bias.data.copy_(flow.tensor(key_b))

            self.value = nn.Linear(hidden_size, self.all_head_size)
            self.value.weight.data.copy_(flow.tensor(value_w.transpose()))
            self.value.bias.data.copy_(flow.tensor(value_b))

            self.dropout = nn.Dropout(hidden_dropout_prob)

        def transpose_for_scores(self, x):
            x = flow.reshape(
                x,
                [-1, self.seq_len, self.num_attention_heads, self.attention_head_size],
            )
            return x.permute(0, 2, 1, 3)

        def forward(self, hidden_states, attention_mask, mid_res):
            query_layer = self.transpose_for_scores(self.query(hidden_states))
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))

            assert np.allclose(
                query_layer.numpy(), mid_res[0], rtol=1e-4, atol=1e-4
            ), "middle query is not correct"

            attention_scores = flow.matmul(query_layer, key_layer.transpose(-2, -1))
            attention_scores = attention_scores / math.sqrt(self.attention_head_size)

            attention_mask = flow.reshape(
                attention_mask, [-1, 1, self.seq_len, self.seq_len]
            )
            attention_mask = flow.cast(attention_mask, dtype=flow.float32)
            addr_blob = (attention_mask - 1.0) * 10000.0
            attention_scores = attention_scores + addr_blob

            # Normalize the attention scores to probabilities.
            attention_probs = flow.softmax(attention_scores, dim=-1)

            # This is actually dropping out entire tokens to attend to, which might
            # seem a bit unusual, but is taken from the original Transformer paper.
            attention_probs = self.dropout(attention_probs)

            context_layer = flow.matmul(attention_probs, value_layer)

            context_layer = context_layer.permute(0, 2, 1, 3)
            context_layer = flow.reshape(
                context_layer, [-1, self.seq_len, self.all_head_size]
            )
            return context_layer

    # eager attention
    B, F, T, N, H = 4, 12, 12, 4, 16
    print("reading saved pkl")
    pickle_file = open("rst_param_cache.pkl", "rb")
    [input, lazy_res, params, mid_res] = pickle.load(pickle_file)
    from_blob, to_blob, mask_blob = input

    eager_self_attention = BertSelfAttention(N, N * H, F, params)
    eager_res = eager_self_attention(
        flow.tensor(from_blob, dtype=flow.float32),
        flow.tensor(mask_blob, dtype=flow.float32),
        mid_res,
    )

    eager_res = eager_res.reshape(-1, N * H).numpy()
    assert np.allclose(lazy_res, eager_res, rtol=1e-4, atol=1e-4)


if __name__ == "__main__":
    args = _parse_args()
    if args.mode == "lazy":
        run_lazy()
    elif args.mode == "eager":
        run_eager()
    else:
        raise NotImplementedError
