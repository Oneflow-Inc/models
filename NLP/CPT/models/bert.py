# referemce to transformers bert model
import oneflow as flow

import oneflow.nn as nn

import math
from typing import Tuple, Optional
from .bert_utils import (
    init_weights,
    find_pruneable_heads_and_indices,
    prune_linear_layer,
    apply_chunking_to_forward,
    position_scores,  # replace einsum
)

ACT2FN = {
    "relu": flow.nn.functional.relu,
    # "silu": silu,
    # "swish": silu,
    "gelu": flow.nn.functional.gelu,
    "tanh": flow.nn.functional.tanh,
    # "gelu_new": gelu_new,
    # "gelu_fast": gelu_fast,
    # "quick_gelu": quick_gelu,
    # "mish": mish,
    # "linear": linear_act,
    "sigmoid": flow.nn.functional.sigmoid,
}


class BertEmbeddings(nn.Module):
    def __init__(
        self,
        vocab_size,
        max_position_embeddings,
        type_vocab_size,
        hidden_size,
        layer_norm_eps=1e-5,
        dropout=0,
        pad_token_id=0,
        position_embedding_type="absolute",
    ):
        super(BertEmbeddings, self).__init__()
        self.word_embeddings = nn.Embedding(
            vocab_size, hidden_size, padding_idx=pad_token_id
        )
        self.position_embeddings = nn.Embedding(max_position_embeddings, hidden_size)
        self.token_type_embeddings = nn.Embedding(type_vocab_size, hidden_size)

        self.LayerNorm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.dropout = nn.Dropout(dropout)
        # position_ids (1, len position emb) is contiguous in memory and exported when serialized
        self.position_embedding_type = position_embedding_type
        self.register_buffer(
            "position_ids", flow.arange(max_position_embeddings).expand(1, -1)
        )
        self.register_buffer(
            "token_type_ids",
            flow.zeros(
                self.position_ids.size(),
                dtype=flow.int64,
                device=self.position_ids.device,
            ),
            persistent=False,
        )

        self.padding_idx = pad_token_id

    def forward(
        self,
        input_ids,
        token_type_ids=None,
        position_ids=None,
        inputs_embeds=None,
        past_key_values_length=0,
    ):

        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        seq_length = input_shape[1]

        if position_ids is None:
            position_ids = self.position_ids[
                :, past_key_values_length : seq_length + past_key_values_length
            ]

        # Setting the token_type_ids to the registered buffer in constructor where it is all zeros, which usually occurs
        # when its auto-generated, registered buffer helps users when tracing the model without passing token_type_ids, solves
        if token_type_ids is None:
            buffered_token_type_ids = self.token_type_ids[:, :seq_length]
            buffered_token_type_ids_expanded = buffered_token_type_ids.expand(
                input_shape[0], seq_length
            )
            token_type_ids = buffered_token_type_ids_expanded
        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = inputs_embeds + token_type_embeddings
        if self.position_embedding_type == "absolute":
            position_embeddings = self.position_embeddings(position_ids)
            embeddings += position_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class BertSelfAttention(nn.Module):
    def __init__(
        self,
        max_position_embeddings,
        hidden_size,
        nheads,
        dropout=0,
        position_embedding_type="absolute",
        is_decoder=False,
    ):
        super(BertSelfAttention, self).__init__()
        if hidden_size % nheads != 0:
            raise ValueError(
                f"The hidden size ({hidden_size}) is not a multiple of the number of attention "
                f"heads ({nheads})"
            )

        self.num_attention_heads = nheads
        self.attention_head_size = int(hidden_size / nheads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(dropout)
        self.position_embedding_type = position_embedding_type
        if (
            self.position_embedding_type == "relative_key"
            or self.position_embedding_type == "relative_key_query"
        ):
            self.max_position_embeddings = max_position_embeddings
            self.distance_embedding = nn.Embedding(
                2 * max_position_embeddings - 1, self.attention_head_size
            )

        self.is_decoder = is_decoder

    def transpose_for_scores(self, x):
        new_x_shape = tuple(x.size()[:-1]) + (
            self.num_attention_heads,
            self.attention_head_size,
        )
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_value=None,
        output_attentions=False,
    ):
        mixed_query_layer = self.query(hidden_states)

        # If this is instantiated as a cross-attention module, the keys
        # and values come from an encoder; the attention mask needs to be
        # such that the encoder's padding tokens are not attended to.
        is_cross_attention = encoder_hidden_states is not None

        if is_cross_attention and past_key_value is not None:
            # reuse k,v, cross_attentions
            key_layer = past_key_value[0]
            value_layer = past_key_value[1]
            attention_mask = encoder_attention_mask
        elif is_cross_attention:
            key_layer = self.transpose_for_scores(self.key(encoder_hidden_states))
            value_layer = self.transpose_for_scores(self.value(encoder_hidden_states))
            attention_mask = encoder_attention_mask
        elif past_key_value is not None:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))
            key_layer = flow.cat([past_key_value[0], key_layer], dim=2)
            value_layer = flow.cat([past_key_value[1], value_layer], dim=2)
        else:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))

        query_layer = self.transpose_for_scores(mixed_query_layer)

        if self.is_decoder:
            # if cross_attention save Tuple(flow.Tensor, flow.Tensor) of all cross attention key/value_states.
            # Further calls to cross_attention layer can then reuse all cross-attention
            # key/value_states (first "if" case)
            # if uni-directional self-attention (decoder) save Tuple(flow.Tensor, flow.Tensor) of
            # all previous decoder key/value_states. Further calls to uni-directional self-attention
            # can concat previous decoder key/value_states to current projected key/value_states (third "elif" case)
            # if encoder bi-directional self-attention `past_key_value` is always `None`
            past_key_value = (key_layer, value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = flow.matmul(query_layer, key_layer.transpose(-1, -2))

        if (
            self.position_embedding_type == "relative_key"
            or self.position_embedding_type == "relative_key_query"
        ):
            seq_length = hidden_states.size()[1]
            position_ids_l = flow.arange(
                seq_length, dtype=flow.int64, device=hidden_states.device
            ).view(-1, 1)
            position_ids_r = flow.arange(
                seq_length, dtype=flow.int64, device=hidden_states.device
            ).view(1, -1)
            distance = position_ids_l - position_ids_r
            positional_embedding = self.distance_embedding(
                distance + self.max_position_embeddings - 1
            )
            positional_embedding = positional_embedding.to(
                dtype=query_layer.dtype
            )  # fp16 compatibility

            if self.position_embedding_type == "relative_key":
                relative_position_scores = position_scores(
                    query_layer, positional_embedding
                )
                attention_scores = attention_scores + relative_position_scores
            elif self.position_embedding_type == "relative_key_query":
                relative_position_scores_query = position_scores(
                    query_layer, positional_embedding
                )
                relative_position_scores_key = position_scores(
                    key_layer, positional_embedding
                )
                attention_scores = (
                    attention_scores
                    + relative_position_scores_query
                    + relative_position_scores_key
                )
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = flow.matmul(attention_probs, value_layer)

        # oneflow doesnot support contiguous()
        context_layer = context_layer.permute(0, 2, 1, 3)  # .contiguous()
        new_context_layer_shape = tuple(context_layer.size()[:-2]) + (
            self.all_head_size,
        )
        context_layer = context_layer.view(*new_context_layer_shape)

        outputs = (
            (context_layer, attention_probs) if output_attentions else (context_layer,)
        )

        if self.is_decoder:
            outputs = outputs + (past_key_value,)
        return outputs


class BertSelfOutput(nn.Module):
    def __init__(self, hidden_size, layer_norm_eps=1e-5, dropout=0):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.dropout = nn.Dropout(dropout)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


# To behave as an decoder or Seq2Seq model the model needs to be initialized with the
# is_decoder argument set to True.


class BertAttention(nn.Module):
    def __init__(
        self,
        max_position_embeddings,
        hidden_size,
        nheads,
        layer_norm_eps=1e-5,
        attn_dropout=0,
        hidden_dropout=0,
        position_embedding_type="absolute",
        is_decoder=False,
    ):
        super(BertAttention, self).__init__()
        self.selfattn = BertSelfAttention(
            max_position_embeddings,
            hidden_size,
            nheads,
            attn_dropout,
            position_embedding_type,
            is_decoder,
        )
        self.output = BertSelfOutput(hidden_size, layer_norm_eps, hidden_dropout)
        self.pruned_heads = set()

    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        heads, index = find_pruneable_heads_and_indices(
            heads,
            self.attn.num_attention_heads,
            self.self.attention_head_size,
            self.pruned_heads,
        )

        # Prune linear layers
        self.selfattn.query = prune_linear_layer(self.selfattn.query, index)
        self.selfattn.key = prune_linear_layer(self.selfattn.key, index)
        self.selfattn.value = prune_linear_layer(self.selfattn.value, index)
        self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)

        # Update hyper params and store pruned heads
        self.selfattn.num_attention_heads = self.selfattn.num_attention_heads - len(
            heads
        )
        self.selfattn.all_head_size = (
            self.selfattn.attention_head_size * self.selfattn.num_attention_heads
        )
        self.pruned_heads = self.pruned_heads.union(heads)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_value=None,
        output_attentions=False,
    ):
        self_outputs = self.selfattn(
            hidden_states,
            attention_mask,
            head_mask,
            encoder_hidden_states,
            encoder_attention_mask,
            past_key_value,
            output_attentions,
        )
        attention_output = self.output(self_outputs[0], hidden_states)
        # add attentions if we output them
        outputs = (attention_output,) + self_outputs[1:]
        return outputs


class BertIntermediate(nn.Module):
    def __init__(self, hidden_size, intermediate_size, activation):
        super(BertIntermediate, self).__init__()
        self.dense = nn.Linear(hidden_size, intermediate_size)
        if isinstance(activation, str):
            self.intermediate_act_fn = ACT2FN[activation]
        else:
            self.intermediate_act_fn = activation

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class BertOutput(nn.Module):
    def __init__(self, hidden_size, intermediate_size, layer_norm_eps=1e-5, dropout=0):
        super(BertOutput, self).__init__()
        self.dense = nn.Linear(intermediate_size, hidden_size)
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.dropout = nn.Dropout(dropout)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


# To behave as an decoder the model needs to be initialized with the
# is_decoder argument set to True.
# To be used in a Seq2Seq model, the model needs to initialized with
# both is_decoder argument and add_cross_attention set to True


class BertLayer(nn.Module):
    def __init__(
        self,
        max_position_embeddings,
        hidden_size,
        intermediate_size,
        nheads,
        activation,
        chunk_size_feed_forward=0,
        layer_norm_eps=1e-5,
        attn_dropout=0,
        hidden_dropout=0,
        position_embedding_type="absolute",
        is_decoder=False,
        add_cross_attention=False,
    ):
        super(BertLayer, self).__init__()
        self.chunk_size_feed_forward = chunk_size_feed_forward
        self.seq_len_dim = 1
        self.attention = BertAttention(
            max_position_embeddings,
            hidden_size,
            nheads,
            layer_norm_eps,
            attn_dropout,
            hidden_dropout,
            position_embedding_type,
            is_decoder,
        )
        self.is_decoder = is_decoder
        self.add_cross_attention = add_cross_attention
        if self.add_cross_attention:
            assert (
                self.is_decoder
            ), f"{self} should be used as a decoder model if cross attention is added"
            self.crossattention = BertAttention(
                max_position_embeddings,
                hidden_size,
                nheads,
                layer_norm_eps,
                attn_dropout,
                hidden_dropout,
                position_embedding_type,
                is_decoder,
            )
        self.intermediate = BertIntermediate(hidden_size, intermediate_size, activation)
        self.output = BertOutput(
            hidden_size, intermediate_size, layer_norm_eps, hidden_dropout
        )

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_value=None,
        output_attentions=False,
    ):
        # decoder uni-directional self-attention cached key/values tuple is at positions 1,2
        self_attn_past_key_value = (
            past_key_value[:2] if past_key_value is not None else None
        )
        self_attention_outputs = self.attention(
            hidden_states,
            attention_mask,
            head_mask,
            None,
            None,
            self_attn_past_key_value,
            output_attentions,
        )
        attention_output = self_attention_outputs[0]

        # if decoder, the last output is tuple of self-attn cache
        if self.is_decoder:
            outputs = self_attention_outputs[1:-1]
            present_key_value = self_attention_outputs[-1]
        else:
            # add self attentions if we output attention weights
            outputs = self_attention_outputs[1:]

        cross_attn_present_key_value = None
        if self.is_decoder and encoder_hidden_states is not None:
            assert hasattr(
                self, "crossattention"
            ), f"If `encoder_hidden_states` are passed, {self} has to be instantiated with cross-attention layers by setting `add_cross_attention=True`"

            # cross_attn cached key/values tuple is at positions 3,4 of past_key_value tuple
            cross_attn_past_key_value = (
                past_key_value[-2:] if past_key_value is not None else None
            )
            cross_attention_outputs = self.crossattention(
                attention_output,
                attention_mask,
                head_mask,
                encoder_hidden_states,
                encoder_attention_mask,
                cross_attn_past_key_value,
                output_attentions,
            )
            attention_output = cross_attention_outputs[0]
            # add cross attentions if we output attention weights
            outputs = outputs + cross_attention_outputs[1:-1]

            # add cross-attn cache to positions 3,4 of present_key_value tuple
            cross_attn_present_key_value = cross_attention_outputs[-1]
            present_key_value = present_key_value + cross_attn_present_key_value

        layer_output = apply_chunking_to_forward(
            self.feed_forward_chunk,
            self.chunk_size_feed_forward,
            self.seq_len_dim,
            attention_output,
        )
        outputs = (layer_output,) + outputs

        # if decoder, return the attn key/values as the last output
        if self.is_decoder:
            outputs = outputs + (present_key_value,)

        return outputs

    def feed_forward_chunk(self, attention_output):
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output


class BertEncoder(nn.Module):
    def __init__(
        self,
        num_layers,
        max_position_embeddings,
        hidden_size,
        intermediate_size,
        nheads,
        activation,
        chunk_size_feed_forward=0,
        layer_norm_eps=1e-5,
        attn_dropout=0,
        hidden_dropout=0,
        position_embedding_type="absolute",
        is_decoder=False,
        add_cross_attention=False,
    ):
        super(BertEncoder, self).__init__()
        self.add_cross_attention = add_cross_attention
        self.num_layers = num_layers

        self.layer = nn.ModuleList(
            [
                BertLayer(
                    max_position_embeddings,
                    hidden_size,
                    intermediate_size,
                    nheads,
                    activation,
                    chunk_size_feed_forward,
                    layer_norm_eps,
                    attn_dropout,
                    hidden_dropout,
                    position_embedding_type,
                    is_decoder,
                    add_cross_attention,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=False,
        output_hidden_states=False,
    ):
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = (
            () if output_attentions and self.add_cross_attention else None
        )

        next_decoder_cache = () if use_cache else None
        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_head_mask = head_mask[i] if head_mask is not None else None
            past_key_value = past_key_values[i] if past_key_values is not None else None

            layer_outputs = layer_module(
                hidden_states,
                attention_mask,
                layer_head_mask,
                encoder_hidden_states,
                encoder_attention_mask,
                past_key_value,
                output_attentions,
            )

            hidden_states = layer_outputs[0]
            if use_cache:
                next_decoder_cache += (layer_outputs[-1],)
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)
                if self.add_cross_attention:
                    all_cross_attentions = all_cross_attentions + (layer_outputs[2],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        # output, past_key_values, hidden_states, attentions, cross_attentions.
        return (
            hidden_states,
            next_decoder_cache,
            all_hidden_states,
            all_self_attentions,
            all_cross_attentions,
        )


class BertPooler(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class Bert(nn.Module):
    def __init__(
        self,
        vocab_size=30522,
        type_vocab_size=2,
        max_position_embeddings=512,
        hidden_size=768,
        intermediate_size=3072,
        chunk_size_feed_forward=0,
        num_layers=12,
        nheads=12,
        activation="gelu",
        pad_token_id=1,
        layer_norm_eps=1e-12,
        attn_dropout=0.1,
        hidden_dropout=0.1,
        position_embedding_type="absolute",
        is_decoder=False,
        add_pooling_layer=True,
        add_cross_attention=False,
    ):
        super(Bert, self).__init__()

        self.embeddings = BertEmbeddings(
            vocab_size,
            max_position_embeddings,
            type_vocab_size,
            hidden_size,
            layer_norm_eps,
            hidden_dropout,
            pad_token_id,
            position_embedding_type,
        )
        self.encoder = BertEncoder(
            num_layers,
            max_position_embeddings,
            hidden_size,
            intermediate_size,
            nheads,
            activation,
            chunk_size_feed_forward,
            layer_norm_eps,
            attn_dropout,
            hidden_dropout,
            position_embedding_type,
            is_decoder,
            add_cross_attention,
        )

        self.pooler = BertPooler(hidden_size) if add_pooling_layer else None
        self.is_decoder = is_decoder
        self.num_layers = num_layers
        self.hidden_size = hidden_size

        self.init_weights()

    def init_weights(self):

        self.apply(init_weights)

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def _prune_heads(self, heads_to_prune):

        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    def get_head_mask(
        self,
        head_mask: Optional[flow.Tensor],
        num_hidden_layers: int,
        is_attention_chunked: bool = False,
    ) -> flow.Tensor:

        if head_mask is not None:
            # convert to 5D tensor
            if head_mask.dim() == 1:
                head_mask = (
                    head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
                )
                head_mask = head_mask.expand(num_hidden_layers, -1, -1, -1, -1)
            elif head_mask.dim() == 2:
                # We can specify head_mask for each layer
                head_mask = head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)
            assert (
                head_mask.dim() == 5
            ), f"head_mask.dim != 5, instead {head_mask.dim()}"
            # switch to float if need + fp16 compatibility
            head_mask = head_mask.to(flow.float)
            if is_attention_chunked is True:
                head_mask = head_mask.unsqueeze(-1)
        else:
            head_mask = [None] * num_hidden_layers

        return head_mask

    def get_extended_attention_mask(
        self, attention_mask: flow.Tensor, input_shape: Tuple[int], device: flow.device
    ) -> flow.Tensor:

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        if attention_mask.dim() == 3:
            extended_attention_mask = attention_mask[:, None, :, :]
        elif attention_mask.dim() == 2:
            # Provided a padding mask of dimensions [batch_size, seq_length]
            # - if the model is a decoder, apply a causal mask in addition to the padding mask
            # - if the model is an encoder, make the mask broadcastable to [batch_size, num_heads, seq_length, seq_length]
            if self.is_decoder:
                batch_size, seq_length = input_shape
                seq_ids = flow.arange(seq_length, device=device)
                causal_mask = (
                    seq_ids[None, None, :].repeat(batch_size, seq_length, 1)
                    <= seq_ids[None, :, None]
                )
                # in case past_key_values are used we need to add a prefix ones mask to the causal mask
                causal_mask = causal_mask.to(attention_mask.dtype)

                if causal_mask.shape[1] < attention_mask.shape[1]:
                    prefix_seq_len = attention_mask.shape[1] - causal_mask.shape[1]
                    causal_mask = flow.cat(
                        [
                            flow.ones(
                                (batch_size, seq_length, prefix_seq_len),
                                device=device,
                                dtype=causal_mask.dtype,
                            ),
                            causal_mask,
                        ],
                        axis=-1,
                    )

                extended_attention_mask = (
                    causal_mask[:, None, :, :] * attention_mask[:, None, None, :]
                )
            else:
                extended_attention_mask = attention_mask[:, None, None, :]
        else:
            raise ValueError(
                f"Wrong shape for input_ids (shape {input_shape}) or attention_mask (shape {attention_mask.shape})"
            )

        extended_attention_mask = extended_attention_mask.to(dtype=flow.float)
        extended_attention_mask = (1.0 - extended_attention_mask) * -1e9
        return extended_attention_mask

    def forward(
        self,
        input_ids,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=False,
        output_hidden_states=False,
    ):

        if not self.is_decoder:
            use_cache = False

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time"
            )
        elif input_ids is not None:
            input_shape = input_ids.size()
            batch_size, seq_length = input_shape
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
            batch_size, seq_length = input_shape
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        # past_key_values_length
        past_key_values_length = (
            past_key_values[0][0].shape[2] if past_key_values is not None else 0
        )

        if attention_mask is None:
            attention_mask = flow.ones(
                ((batch_size, seq_length + past_key_values_length)), device=device
            )
        if token_type_ids is None:
            if hasattr(self.embeddings, "token_type_ids"):
                buffered_token_type_ids = self.embeddings.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(
                    batch_size, seq_length
                )
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = flow.zeros(
                    input_shape, dtype=flow.int64, device=device
                )

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask: flow.Tensor = self.get_extended_attention_mask(
            attention_mask, input_shape, device
        )
        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.is_decoder and encoder_hidden_states is not None:
            (
                encoder_batch_size,
                encoder_sequence_length,
                _,
            ) = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = flow.ones(encoder_hidden_shape, device=device)
            encoder_extended_attention_mask = self.invert_attention_mask(
                encoder_attention_mask
            )
        else:
            encoder_extended_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_layers x num_heads]
        # and head_mask is converted to shape [num_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.num_layers)
        embedding_output = self.embeddings(
            input_ids,
            token_type_ids,
            position_ids,
            inputs_embeds,
            past_key_values_length,
        )
        encoder_outputs = self.encoder(
            embedding_output,
            extended_attention_mask,
            head_mask,
            encoder_hidden_states,
            encoder_extended_attention_mask,
            past_key_values,
            use_cache,
            output_attentions,
            output_hidden_states,
        )
        sequence_output = encoder_outputs[0]
        pooled_output = (
            self.pooler(sequence_output) if self.pooler is not None else None
        )

        # sequence_output, pooled_output, past_key_values, hidden_states, attentions, cross_attentions.
        return (sequence_output, pooled_output) + encoder_outputs[1:]
