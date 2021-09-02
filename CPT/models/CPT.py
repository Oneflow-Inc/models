from logging import log
import math
import random
from typing import Optional, Tuple

import oneflow as flow
import oneflow.nn as nn
from oneflow.nn import CrossEntropyLoss, MSELoss

from .bert import Bert
from .bart_utils import (
    shift_tokens_right,
    _make_causal_mask,
    _expand_mask,
    init_weights,
    tensor_unique,  # for tensor.unique
)
# fix LayerNorm bugs
from .dev_ops import LayerNorm

ACT2FN = {
    "relu": flow.F.relu,
    # "silu": silu,
    # "swish": silu,
    "gelu": flow.F.gelu,
    "tanh": flow.tanh,
    # "gelu_new": gelu_new,
    # "gelu_fast": gelu_fast,
    # "quick_gelu": quick_gelu,
    # "mish": mish,
    # "linear": linear_act,
    "sigmoid": flow.sigmoid,
}


class BartLearnedPositionalEmbedding(nn.Embedding):
    """
    This module learns positional embeddings up to a fixed maximum size.
    """

    def __init__(self, num_embeddings: int, embedding_dim: int):
        # Bart is set up so that if padding_idx is specified then offset the embedding ids by 2
        # and adjust num_embeddings appropriately. Other models dont have this hack
        self.offset = 2
        super().__init__(num_embeddings + self.offset, embedding_dim)

    def forward(self, input_ids_shape: flow.Size, past_key_values_length: int = 0):
        """`input_ids_shape` is expected to be [bsz x seqlen]."""
        bsz, seq_len = input_ids_shape[:2]
        positions = flow.arange(
            past_key_values_length, past_key_values_length + seq_len, dtype=flow.long, device=self.weight.device
        )
        return super().forward(positions + self.offset)


class BartAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        is_decoder: bool = False,
        bias: bool = True,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        assert (
            self.head_dim * num_heads == self.embed_dim
        ), f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`: {num_heads})."
        self.scaling = self.head_dim ** -0.5
        self.is_decoder = is_decoder

        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

    def _shape(self, tensor: flow.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

    def forward(
        self,
        hidden_states: flow.Tensor,
        key_value_states: Optional[flow.Tensor] = None,
        past_key_value: Optional[Tuple[flow.Tensor]] = None,
        attention_mask: Optional[flow.Tensor] = None,
        layer_head_mask: Optional[flow.Tensor] = None,
        output_attentions: bool = False,
    ) -> Tuple[flow.Tensor, Optional[flow.Tensor], Optional[Tuple[flow.Tensor]]]:
        """Input shape: Batch x Time x Channel"""

        # if key_value_states are provided this layer is used as a cross-attention layer
        # for the decoder
        is_cross_attention = key_value_states is not None
        bsz, tgt_len, embed_dim = hidden_states.size()

        # get query proj
        query_states = self.q_proj(hidden_states) * self.scaling
        # get key, value proj
        if is_cross_attention and past_key_value is not None:
            # reuse k,v, cross_attentions
            key_states = past_key_value[0]
            value_states = past_key_value[1]
        elif is_cross_attention:
            # cross_attentions
            key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
            value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
        elif past_key_value is not None:
            # reuse k, v, self_attention
            key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
            value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
            key_states = flow.cat([past_key_value[0], key_states], dim=2)
            value_states = flow.cat([past_key_value[1], value_states], dim=2)
        else:
            # self_attention
            key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
            value_states = self._shape(self.v_proj(hidden_states), -1, bsz)

        if self.is_decoder:
            # if cross_attention save Tuple(flow.Tensor, flow.Tensor) of all cross attention key/value_states.
            # Further calls to cross_attention layer can then reuse all cross-attention
            # key/value_states (first "if" case)
            # if uni-directional self-attention (decoder) save Tuple(flow.Tensor, flow.Tensor) of
            # all previous decoder key/value_states. Further calls to uni-directional self-attention
            # can concat previous decoder key/value_states to current projected key/value_states (third "elif" case)
            # if encoder bi-directional self-attention `past_key_value` is always `None`
            past_key_value = (key_states, value_states)

        proj_shape = (bsz * self.num_heads, -1, self.head_dim)
        query_states = self._shape(
            query_states, tgt_len, bsz).view(*proj_shape)
        key_states = key_states.view(*proj_shape)
        value_states = value_states.view(*proj_shape)

        src_len = key_states.size(1)
        attn_weights = flow.bmm(query_states, key_states.transpose(1, 2))

        assert attn_weights.size() == (
            bsz * self.num_heads,
            tgt_len,
            src_len,
        ), f"Attention weights should be of size {(bsz * self.num_heads, tgt_len, src_len)}, but is {attn_weights.size()}"

        if attention_mask is not None:
            assert attention_mask.size() == (
                bsz,
                1,
                tgt_len,
                src_len,
            ), f"Attention mask should be of size {(bsz, 1, tgt_len, src_len)}, but is {attention_mask.size()}"
            attn_weights = attn_weights.view(
                bsz, self.num_heads, tgt_len, src_len) + attention_mask
            attn_weights = attn_weights.view(
                bsz * self.num_heads, tgt_len, src_len)

        attn_weights = attn_weights.softmax(dim=-1)

        if layer_head_mask is not None:
            assert layer_head_mask.size() == (
                self.num_heads,
            ), f"Head mask for a single layer should be of size {(self.num_heads,)}, but is {layer_head_mask.size()}"
            attn_weights = layer_head_mask.view(
                1, -1, 1, 1) * attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights.view(
                bsz * self.num_heads, tgt_len, src_len)

        if output_attentions:
            # this operation is a bit akward, but it's required to
            # make sure that attn_weights keeps its gradient.
            # In order to do so, attn_weights have to reshaped
            # twice and have to be reused in the following
            attn_weights_reshaped = attn_weights.view(
                bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights_reshaped.view(
                bsz * self.num_heads, tgt_len, src_len)
        else:
            attn_weights_reshaped = None

        # with mpu.get_cuda_rng_tracker().fork():
        # prob = self.dropout if self.training else 0
        # attn_probs = flow.F.dropout(attn_weights, p=prob)    
        # attn_output = flow.bmm(attn_probs, value_states)
        if self.training:
            attn_weights = flow.F.dropout(attn_weights, p=self.dropout)
        attn_output = flow.bmm(attn_weights, value_states)

        assert attn_output.size() == (
            bsz * self.num_heads,
            tgt_len,
            self.head_dim,
        ), f"`attn_output` should be of size {(bsz, self.num_heads, tgt_len, self.head_dim)}, but is {attn_output.size()}"

        attn_output = (
            attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
            .transpose(1, 2)
            .reshape(bsz, tgt_len, embed_dim)
        )

        attn_output = self.out_proj(attn_output)

        return attn_output, attn_weights_reshaped, past_key_value


class BartDecoderLayer(nn.Module):
    def __init__(self, d_model: int = 1024, num_heads: int = 16, ffn_dim: int = 4096, activation: str = "gelu",
                 attn_dropout: float = 0.0, hidden_dropout: float = 0.0, act_dropout: float = 0.0):
        super(BartDecoderLayer, self).__init__()
        self.embed_dim = d_model

        self.self_attn = BartAttention(
            embed_dim=self.embed_dim,
            num_heads=num_heads,
            dropout=attn_dropout,
            is_decoder=True,
        )
        self.dropout = hidden_dropout
        self.activation_fn = ACT2FN[activation]
        self.activation_dropout = act_dropout

        self.self_attn_layer_norm = LayerNorm(self.embed_dim)
        self.encoder_attn = BartAttention(
            self.embed_dim,
            num_heads,
            dropout=attn_dropout,
            is_decoder=True,
        )
        self.encoder_attn_layer_norm = LayerNorm(self.embed_dim)
        self.fc1 = nn.Linear(self.embed_dim, ffn_dim)
        self.fc2 = nn.Linear(ffn_dim, self.embed_dim)
        self.final_layer_norm = LayerNorm(self.embed_dim)

    def forward(
        self,
        hidden_states: flow.Tensor,
        attention_mask: Optional[flow.Tensor] = None,
        encoder_hidden_states: Optional[flow.Tensor] = None,
        encoder_attention_mask: Optional[flow.Tensor] = None,
        layer_head_mask: Optional[flow.Tensor] = None,
        encoder_layer_head_mask: Optional[flow.Tensor] = None,
        past_key_value: Optional[Tuple[flow.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = True,
    ):
        residual = hidden_states

        # Self Attention
        # decoder uni-directional self-attention cached key/values tuple is at positions 1,2
        self_attn_past_key_value = past_key_value[:
                                                  2] if past_key_value is not None else None
        # add present self-attn cache to positions 1,2 of present_key_value tuple
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states, None, self_attn_past_key_value, attention_mask, layer_head_mask, output_attentions)
        if self.training:
            hidden_states = flow.F.dropout(hidden_states, p=self.dropout)
        hidden_states = residual + hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)

        # Cross-Attention Block
        cross_attn_present_key_value = None
        cross_attn_weights = None
        if encoder_hidden_states is not None:
            residual = hidden_states

            # cross_attn cached key/values tuple is at positions 3,4 of present_key_value tuple
            cross_attn_past_key_value = past_key_value[-2:
                                                       ] if past_key_value is not None else None
            hidden_states, cross_attn_weights, cross_attn_present_key_value = self.encoder_attn(
                hidden_states, encoder_hidden_states, cross_attn_past_key_value, encoder_attention_mask, encoder_layer_head_mask, output_attentions)
            if self.training:
                hidden_states = flow.F.dropout(hidden_states, p=self.dropout)
            hidden_states = residual + hidden_states
            hidden_states = self.encoder_attn_layer_norm(hidden_states)

            # add cross-attn to positions 3,4 of present_key_value tuple
            present_key_value = present_key_value + cross_attn_present_key_value

        # Fully Connected
        residual = hidden_states
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        if self.training:
            hidden_states = flow.F.dropout(
                hidden_states, p=self.activation_dropout)
        hidden_states = self.fc2(hidden_states)
        if self.training:
            hidden_states = flow.F.dropout(hidden_states, p=self.dropout)
        hidden_states = residual + hidden_states
        hidden_states = self.final_layer_norm(hidden_states)

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights, cross_attn_weights)

        if use_cache:
            outputs += (present_key_value,)

        return outputs


class BartDecoder(nn.Module):
    """
    Transformer decoder consisting of *num_layers* layers. Each layer is a :class:`BartDecoderLayer`

    Args:
        embed_tokens (flow.nn.Embedding): output embedding
    """

    def __init__(self, d_model: int = 1024, vocab_size: int = 50265, num_layers: int = 12,
                 decoder_attn_heads: int = 16, decoder_ffn_dim: int = 4096,
                 max_position_embeddings: int = 1024, activation="gelu", pad_token_id: int = 1,
                 attn_dropout: float = 0.0, hidden_dropout: float = 0.0, act_dropout=0.0, decoder_layerdrop: float = 0.0,
                 scale_embedding: bool = False, embed_tokens: Optional[nn.Embedding] = None):
        super(BartDecoder, self).__init__()
        self.dropout = hidden_dropout
        self.layerdrop = decoder_layerdrop
        self.padding_idx = pad_token_id
        self.max_target_positions = max_position_embeddings
        self.embed_scale = math.sqrt(
            d_model) if scale_embedding else 1.0
        self.num_layers = num_layers

        if embed_tokens is not None:
            self.embed_tokens = embed_tokens
        else:
            self.embed_tokens = nn.Embedding(
                vocab_size, d_model, self.padding_idx)

        self.embed_positions = BartLearnedPositionalEmbedding(
            max_position_embeddings, d_model)
        self.layers = nn.ModuleList(
            [BartDecoderLayer(d_model, decoder_attn_heads, decoder_ffn_dim, activation, attn_dropout, hidden_dropout, act_dropout)
             for _ in range(num_layers)])
        self.layernorm_embedding = LayerNorm(d_model)

        self.init_weights()

    def init_weights(self):

        self.apply(init_weights)

    def _prepare_decoder_attention_mask(self, attention_mask, input_shape, inputs_embeds, past_key_values_length):
        # create causal mask
        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        combined_attention_mask = None
        if input_shape[-1] > 1:
            combined_attention_mask = _make_causal_mask(
                input_shape, inputs_embeds.dtype, inputs_embeds.device, past_key_values_length=past_key_values_length
            )

        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            expanded_attn_mask = _expand_mask(
                attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1])
            combined_attention_mask = (
                expanded_attn_mask if combined_attention_mask is None else expanded_attn_mask +
                combined_attention_mask
            )

        return combined_attention_mask

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        head_mask=None,
        encoder_head_mask=None,
        past_key_values=None,
        inputs_embeds=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
    ):
        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError(
                "You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError(
                "You have to specify either decoder_input_ids or decoder_inputs_embeds")

        # past_key_values_length
        past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids) * self.embed_scale

        attention_mask = self._prepare_decoder_attention_mask(
            attention_mask, input_shape, inputs_embeds, past_key_values_length
        )

        # expand encoder attention mask
        if encoder_hidden_states is not None and encoder_attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            encoder_attention_mask = _expand_mask(
                encoder_attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1])

        # embed positions
        positions = self.embed_positions(input_shape, past_key_values_length)

        hidden_states = inputs_embeds + positions
        hidden_states = self.layernorm_embedding(hidden_states)

        if self.training:
            hidden_states = flow.F.dropout(hidden_states, p=self.dropout)

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        all_cross_attentions = () if (
            output_attentions and encoder_hidden_states is not None) else None
        next_decoder_cache = () if use_cache else None

        # check if head_mask has a correct number of layers specified if desired
        if head_mask is not None:
            assert head_mask.size()[0] == (
                len(self.layers)
            ), f"The head_mask should be specified for {len(self.layers)} layers, but it is for {head_mask.size()[0]}."
        for idx, decoder_layer in enumerate(self.layers):
            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            dropout_probability = random.uniform(0, 1)
            if self.training and (dropout_probability < self.layerdrop):
                continue

            past_key_value = past_key_values[idx] if past_key_values is not None else None

            layer_outputs = decoder_layer(hidden_states, attention_mask, encoder_hidden_states, encoder_attention_mask, (head_mask[idx] if head_mask is not None else None), (
                encoder_head_mask[idx] if encoder_head_mask is not None else None), past_key_value, output_attentions, use_cache)
            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache += (
                    layer_outputs[3 if output_attentions else 1],)

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

                if encoder_hidden_states is not None:
                    all_cross_attentions += (layer_outputs[2],)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None

        # last_hidden_state, past_key_value, hidden_states, attentions, cross_attentions
        return hidden_states, next_cache, all_hidden_states, all_self_attns, all_cross_attentions


class CPT(nn.Module):
    def __init__(self, d_model: int = 1024, vocab_size: int = 50265,
                 num_encoder_layers: int = 12, num_decoder_layers: int = 12,
                 encoder_attn_heads: int = 16, decoder_attn_heads: int = 16,
                 encoder_ffn_dim: int = 4096, decoder_ffn_dim: int = 4096,
                 max_position_embeddings: int = 1024, activation="gelu", pad_token_id: int = 1,
                 attn_dropout: float = 0.0, hidden_dropout: float = 0.0, act_dropout=0.0,
                 decoder_layerdrop: float = 0.0, scale_embedding: bool = False,
                 decoder_start_token_id=2, encoder_layernorm_eps=1e-12):
        super(CPT, self).__init__()
        self.encoder = Bert(vocab_size=vocab_size, hidden_size=d_model, num_layers=num_encoder_layers, nheads=encoder_attn_heads,
                            intermediate_size=encoder_ffn_dim, hidden_dropout=act_dropout, attn_dropout=act_dropout, add_pooling_layer=False,
                            layer_norm_eps=encoder_layernorm_eps)
        self.shared = self.encoder.get_input_embeddings()
        self.decoder = BartDecoder(d_model, vocab_size, num_decoder_layers, decoder_attn_heads, decoder_ffn_dim, max_position_embeddings,
                                   activation, pad_token_id, attn_dropout, hidden_dropout, act_dropout, decoder_layerdrop, scale_embedding, self.shared)
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.num_decoder_layers = num_decoder_layers
        self.num_encoder_layers = num_encoder_layers
        self.pad_token_id = pad_token_id
        self.decoder_start_token_id = decoder_start_token_id
        self.init_weights()

    def init_weights(self):
        self.apply(init_weights)

    def get_input_embeddings(self):
        return self.shared

    def set_input_embeddings(self, value):
        self.shared = value
        self.encoder.embed_tokens = self.shared
        self.decoder.embed_tokens = self.shared

    def get_encoder(self):
        class _Encoder(flow.nn.Module):
            def __init__(self, encoder):
                super().__init__()
                self.encoder = encoder
            
            def forward(self, *args, **kwargs):
                kwargs['output_hidden_states'] = True                
                return self.encoder(*args, **kwargs)
        return _Encoder(self.encoder)

    def get_decoder(self):
        return self.decoder

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        encoder_outputs=None,
        past_key_values=None,
        inputs_embeds=None,
        decoder_inputs_embeds=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
    ):

        # different to other models, Bart automatically creates decoder_input_ids from
        # input_ids if no decoder_input_ids are provided
        if decoder_input_ids is None and decoder_inputs_embeds is None:
            decoder_input_ids = shift_tokens_right(
                input_ids, self.pad_token_id, self.decoder_start_token_id
            )

        if encoder_outputs is None:
            encoder_outputs = self.encoder(input_ids, attention_mask, flow.ones_like(
                input_ids), None, head_mask, inputs_embeds, None, None, None, None, output_attentions, True)
            # last_hidden_states, hidden_states, attentions
            encoder_outputs = (encoder_outputs[0], encoder_outputs[3], encoder_outputs[4])
        # If the user passed a tuple for encoder_outputs
        elif isinstance(encoder_outputs, (tuple, list)):
            encoder_outputs = (
                encoder_outputs[0],
                encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )

        if isinstance(encoder_outputs, (flow.Tensor)):
            encoder_hidden_states = encoder_outputs
            encoder_outputs = (encoder_outputs,)
        else:
            encoder_hidden_states = encoder_outputs[1][-self.num_decoder_layers - 1] if encoder_outputs[1] is not None else None


        # decoder outputs consists of (dec_features, past_key_value, dec_hidden, dec_attn)
        decoder_outputs = self.decoder(decoder_input_ids, decoder_attention_mask, encoder_hidden_states, attention_mask,
                                       decoder_head_mask, head_mask, past_key_values, decoder_inputs_embeds, use_cache, output_attentions, output_hidden_states)

        # last_hidden_state, past_key_values, decoder_hidden_states, decoder_attentions, cross_attentnions
        # encoder_last_hidden_state, encoder_hidden_states, encoder_attentions
        return decoder_outputs + encoder_outputs

class BartDecoderWrapper(nn.Module):
    """
    This wrapper class is a helper class to correctly load pretrained checkpoints when the causal language model is
    used in combination with the :class:`~transformers.EncoderDecoderModel` framework.
    """

    def __init__(self, d_model: int = 1024, vocab_size: int = 50265, num_layers: int = 12,
                 decoder_attn_heads: int = 16, decoder_ffn_dim: int = 4096,
                 max_position_embeddings: int = 1024, activation="gelu", pad_token_id: int = 1,
                 attn_dropout: float = 0.0, hidden_dropout: float = 0.0, act_dropout=0.0,
                 decoder_layerdrop: float = 0.0, scale_embedding: bool = False,
                 embed_tokens: Optional[nn.Embedding] = None):
        super(BartDecoderWrapper, self).__init__()
        self.decoder = BartDecoder(d_model, vocab_size, num_layers, decoder_attn_heads, decoder_ffn_dim, max_position_embeddings,
                                   activation, pad_token_id, attn_dropout, hidden_dropout, act_dropout, decoder_layerdrop, scale_embedding, embed_tokens)

    def forward(self, *args, **kwargs):
        return self.decoder(*args, **kwargs)

class BartClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, input_dim: int, inner_dim: int, num_classes: int, pooler_dropout: float):
        super().__init__()
        self.dense = nn.Linear(input_dim, inner_dim)
        self.dropout = nn.Dropout(p=pooler_dropout)
        self.out_proj = nn.Linear(inner_dim, num_classes)

    def forward(self, hidden_states: flow.Tensor):
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.dense(hidden_states)
        hidden_states = flow.tanh(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.out_proj(hidden_states)
        return hidden_states

class CPTForCausalLM(nn.Module):

    def __init__(self, d_model: int = 1024, vocab_size: int = 50265, num_layers: int = 12,
                 decoder_attn_heads: int = 16, decoder_ffn_dim: int = 4096,
                 max_position_embeddings: int = 1024, activation="gelu", pad_token_id: int = 1,
                 attn_dropout: float = 0.0, hidden_dropout: float = 0.0, act_dropout=0.0,
                 decoder_layerdrop: float = 0.0, scale_embedding: bool = False,
                 embed_tokens: Optional[nn.Embedding] = None):
        super(CPTForCausalLM, self).__init__()

        self.model = BartDecoderWrapper(d_model, vocab_size, num_layers, decoder_attn_heads, decoder_ffn_dim, max_position_embeddings,
                                        activation, pad_token_id, attn_dropout, hidden_dropout, act_dropout, decoder_layerdrop, scale_embedding, embed_tokens)

        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        self.vocab_size = vocab_size

        self.init_weights()

    def init_weights(self):

        self.apply(init_weights)

    def get_input_embeddings(self):
        return self.model.decoder.embed_tokens

    def set_input_embeddings(self, value):
        self.model.decoder.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model.decoder = decoder

    def get_decoder(self):
        return self.model.decoder

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        head_mask=None,
        encoder_head_mask=None,
        past_key_values=None,
        inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
    ):

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model.decoder(input_ids, attention_mask, encoder_hidden_states, encoder_attention_mask, head_mask,
                                     encoder_head_mask, past_key_values, inputs_embeds, use_cache, output_attentions, output_hidden_states)
        logits = self.lm_head(outputs[0])

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(
                logits.view(-1, self.vocab_size), labels.view(-1))

        return (loss, logits) + outputs[1:]

    def prepare_inputs_for_generation(self, input_ids, past=None, attention_mask=None, use_cache=None, **kwargs):
        # if model is used as a decoder in encoder-decoder model, the decoder attention mask is created on the fly
        if attention_mask is None:
            attention_mask = input_ids.new_ones(input_ids.shape)

        if past:
            input_ids = input_ids[:, -1:]
        # first step, decoder_cached_states are empty
        return {
            "input_ids": input_ids,  # encoder_outputs is defined. input_ids not needed
            "attention_mask": attention_mask,
            "past_key_values": past,
            "use_cache": use_cache,
        }

    @staticmethod
    def _reorder_cache(past, beam_idx):
        reordered_past = ()
        for layer_past in past:
            reordered_past += (tuple(past_state.index_select(0, beam_idx)
                               for past_state in layer_past),)
        return reordered_past


class CPTForMaskedLM(nn.Module):
    def __init__(self, cls_mode: int = 2, d_model: int = 1024, vocab_size: int = 50265,
                 num_encoder_layers: int = 12, num_decoder_layers: int = 12,
                 encoder_attn_heads: int = 16, decoder_attn_heads: int = 16,
                 encoder_ffn_dim: int = 4096, decoder_ffn_dim: int = 4096,
                 max_position_embeddings: int = 1024, activation="gelu",
                 pad_token_id: int = 1, attn_dropout: float = 0.0,
                 hidden_dropout: float = 0.0, act_dropout=0.0,
                 decoder_layerdrop: float = 0.0, scale_embedding: bool = False, 
                 decoder_start_token_id=2):
        super(CPTForMaskedLM, self).__init__()
        self.model = CPT(d_model, vocab_size, num_encoder_layers, num_decoder_layers, encoder_attn_heads, decoder_attn_heads, encoder_ffn_dim, decoder_ffn_dim, max_position_embeddings,
                          activation, pad_token_id, attn_dropout, hidden_dropout, act_dropout, decoder_layerdrop, scale_embedding, decoder_start_token_id)
        self.cls_mode = cls_mode

        self.register_buffer("final_logits_bias", flow.zeros((1, self.model.shared.num_embeddings)))
        self.lm_head = nn.Linear(d_model, self.model.shared.num_embeddings, bias=False)

        self.init_weights()

    def init_weights(self):

        self.apply(init_weights)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        encoder_outputs=None,
        inputs_embeds=None,
        decoder_inputs_embeds=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
    ):   
        if input_ids is None and inputs_embeds is not None:
            raise NotImplementedError(
                f"Passing input embeddings is currently not supported for {self.__class__.__name__}"
            )
            
        outputs = self.model(input_ids, attention_mask, decoder_input_ids, decoder_attention_mask, head_mask, decoder_head_mask,
                             encoder_outputs, None, inputs_embeds, decoder_inputs_embeds, use_cache, output_attentions, output_hidden_states)

        hidden_states = outputs[0]
        enc_hidden_states = outputs[5]

        dec_logits = self.lm_head(hidden_states) + self.final_logits_bias
        enc_logits = self.lm_head(enc_hidden_states) + self.final_logits_bias

        return (enc_logits, dec_logits) + outputs[1:]



class CPTForConditionalGeneration(nn.Module):
    def __init__(self, d_model: int = 1024, vocab_size: int = 50265,
                 num_encoder_layers: int = 12, num_decoder_layers: int = 12,
                 encoder_attn_heads: int = 16, decoder_attn_heads: int = 16,
                 encoder_ffn_dim: int = 4096, decoder_ffn_dim: int = 4096,
                 max_position_embeddings: int = 1024, activation="gelu", pad_token_id: int = 1,
                 attn_dropout: float = 0.0, hidden_dropout: float = 0.0, act_dropout=0.0,
                 decoder_layerdrop: float = 0.0, scale_embedding: bool = False,
                 decoder_start_token_id=2):
        super(CPTForConditionalGeneration, self).__init__()
        self.model = CPT(d_model, vocab_size, num_encoder_layers, num_decoder_layers, encoder_attn_heads, decoder_attn_heads, encoder_ffn_dim, decoder_ffn_dim, max_position_embeddings,
                          activation, pad_token_id, attn_dropout, hidden_dropout, act_dropout, decoder_layerdrop, scale_embedding, decoder_start_token_id)
        self.register_buffer("final_logits_bias", flow.zeros(
            (1, self.model.shared.num_embeddings)))
        self.lm_head = nn.Linear(
            d_model, self.model.shared.num_embeddings, bias=False)

        self.pad_token_id = pad_token_id
        self.decoder_start_token_id = decoder_start_token_id
        self.vocab_size = vocab_size

        self.init_weights()

    def init_weights(self):

        self.apply(init_weights)

    def get_encoder(self):
        return self.model.get_encoder()

    def get_decoder(self):
        return self.model.get_decoder()

    def resize_token_embeddings(self, new_num_tokens: int) -> nn.Embedding:
        new_embeddings = super().resize_token_embeddings(new_num_tokens)
        self._resize_final_logits_bias(new_num_tokens)
        return new_embeddings

    def _resize_final_logits_bias(self, new_num_tokens: int) -> None:
        old_num_tokens = self.final_logits_bias.shape[-1]
        if new_num_tokens <= old_num_tokens:
            new_bias = self.final_logits_bias[:, :new_num_tokens]
        else:
            extra_bias = flow.zeros(
                (1, new_num_tokens - old_num_tokens), device=self.final_logits_bias.device)
            new_bias = flow.cat([self.final_logits_bias, extra_bias], dim=1)
        self.register_buffer("final_logits_bias", new_bias)

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        encoder_outputs=None,
        past_key_values=None,
        inputs_embeds=None,
        decoder_inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
    ):

        if labels is not None:
            if decoder_input_ids is None:
                decoder_input_ids = shift_tokens_right(
                    labels, self.pad_token_id, self.decoder_start_token_id
                )

        outputs = self.model(input_ids, attention_mask, decoder_input_ids, decoder_attention_mask, head_mask, decoder_head_mask,
                             encoder_outputs, past_key_values, inputs_embeds, decoder_inputs_embeds, use_cache, output_attentions, output_hidden_states)
        
        lm_logits = self.lm_head(outputs[0]) + self.final_logits_bias

        masked_lm_loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            masked_lm_loss = loss_fct(
                lm_logits.view(-1, self.vocab_size), labels.view(-1))

        # loss, logits, past_key_values, decoder_hidden_states, decoder_attentions, cross_attentions
        # encoder_last_hidden_state, encoder_hidden_states, encoder_attentions
        return (masked_lm_loss, lm_logits, ) + outputs[1:]

    def prepare_inputs_for_generation(
        self,
        decoder_input_ids,
        past=None,
        attention_mask=None,
        head_mask=None,
        use_cache=None,
        encoder_outputs=None
    ):
        # cut decoder_input_ids if past is used
        if past is not None:
            decoder_input_ids = decoder_input_ids[:, -1:]

        return {
            "input_ids": None,  # encoder_outputs is defined. input_ids not needed
            "encoder_outputs": encoder_outputs,
            "past_key_values": past,
            "decoder_input_ids": decoder_input_ids,
            "attention_mask": attention_mask,
            "head_mask": head_mask,
            # change this to avoid caching (presumably for debugging)
            "use_cache": use_cache,
        }

    @staticmethod
    def _expand_inputs_for_generation(
        input_ids: flow.Tensor,
        expand_size: int = 1,
        is_encoder_decoder: bool = False,
        attention_mask: flow.Tensor = None,
        encoder_outputs=None,
        **model_kwargs,
    ):
        expanded_return_idx = (flow.arange(input_ids.shape[0]).view(-1, 1).repeat(1, expand_size)
                               .view(-1).to(input_ids.device))
        input_ids = input_ids.index_select(0, expanded_return_idx)

        if "token_type_ids" in model_kwargs:
            token_type_ids = model_kwargs["token_type_ids"]
            model_kwargs["token_type_ids"] = token_type_ids.index_select(
                0, expanded_return_idx)

        if attention_mask is not None:
            model_kwargs["attention_mask"] = attention_mask.index_select(
                0, expanded_return_idx)

        if is_encoder_decoder:
            assert encoder_outputs is not None
            device = encoder_outputs.last_hidden_state.device
            encoder_outputs["hidden_states"] = tuple(h.index_select(0, expanded_return_idx.to(device))
                                                     for h in encoder_outputs["hidden_states"])
            model_kwargs["encoder_outputs"] = encoder_outputs
        return input_ids, model_kwargs

    def prepare_decoder_input_ids_from_labels(self, labels: flow.Tensor):
        return shift_tokens_right(labels, self.pad_token_id, self.decoder_start_token_id)

    @staticmethod
    def _reorder_cache(past, beam_idx):
        reordered_past = ()
        for layer_past in past:
            # cached cross_attention states don't have to be reordered -> they are always the same
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx)
                      for past_state in layer_past[:2]) + layer_past[2:],
            )
        return reordered_past


class CPTForSequenceClassification(nn.Module):
    def __init__(self, cls_mode: int = 2, num_labels: int = 2,
                 d_model: int = 1024, vocab_size: int = 50265,
                 num_encoder_layers: int = 12, num_decoder_layers: int = 12,
                 encoder_attn_heads: int = 16, decoder_attn_heads: int = 16,
                 encoder_ffn_dim: int = 4096, decoder_ffn_dim: int = 4096,
                 max_position_embeddings: int = 1024, activation="gelu",
                 pad_token_id: int = 1, attn_dropout: float = 0.0,
                 hidden_dropout: float = 0.0, act_dropout=0.0,
                 classifier_dropout=0.0, decoder_layerdrop: float = 0.0,
                 scale_embedding: bool = False, decoder_start_token_id=2,
                 eos_token_id=2):
        super(CPTForSequenceClassification, self).__init__()
        self.model = CPT(d_model, vocab_size, num_encoder_layers, num_decoder_layers, encoder_attn_heads, decoder_attn_heads, encoder_ffn_dim, decoder_ffn_dim, max_position_embeddings,
                          activation, pad_token_id, attn_dropout, hidden_dropout, act_dropout, decoder_layerdrop, scale_embedding, decoder_start_token_id)
        # Encoder for classification
        if cls_mode == 1:
            cls_dim = d_model
        # Decoder for classification
        elif cls_mode == 2:
            cls_dim = d_model
        # Both encoder & decoder for classification
        elif cls_mode == 3:
            cls_dim = d_model * 2
        else:
            raise NotImplementedError

        self.cls_head = BartClassificationHead(
            cls_dim,
            cls_dim,
            num_labels,
            classifier_dropout
        )
        init_weights(self.cls_head.dense)
        init_weights(self.cls_head.out_proj)
        self.cls_mode = cls_mode
        self.num_labels = num_labels
        self.eos_token_id = eos_token_id

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        encoder_outputs=None,
        inputs_embeds=None,
        decoder_inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
    ):
        r"""
        labels (:obj:`flow.Tensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the sequence classification/regression loss. Indices should be in :obj:`[0, ...,
            num_labels - 1]`. If :obj:`num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        if labels is not None:
            use_cache = False

        if input_ids is None and inputs_embeds is not None:
            raise NotImplementedError(
                f"Passing input embeddings is currently not supported for {self.__class__.__name__}"
            )
        outputs = self.model(input_ids, attention_mask, decoder_input_ids, decoder_attention_mask, head_mask, decoder_head_mask,
                             encoder_outputs, None, inputs_embeds, decoder_inputs_embeds, use_cache, output_attentions, output_hidden_states)

        hidden_states = outputs[0]
        enc_hidden_states = outputs[5]
        enc_rep = enc_hidden_states[:, 0]

        eos_mask = input_ids.eq(self.eos_token_id).to(flow.int32)

        # flow.unique(eos_mask.sum(1))
        if tensor_unique(eos_mask.sum(1)).shape[0] > 1:
            raise ValueError(
                "All examples must have the same number of <eos> tokens.")
        dec_rep = hidden_states[eos_mask, :].view(hidden_states.size(0), -1, hidden_states.size(-1))[
            :, -1, :
        ]

        if self.cls_mode == 1:
            logits = self.cls_head(enc_rep)
        elif self.cls_mode == 2:
            logits = self.cls_head(dec_rep)
        elif self.cls_mode == 3:
            rep = flow.cat([enc_rep, dec_rep], dim=-1)
            logits = self.cls_head(rep)
        else:
            raise NotImplementedError

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(
                logits.view(-1, self.num_labels), labels.view(-1))

        # loss, logits, past_key_values, decoder_hidden_states, decoder_attentions, cross_attentions,
        # encoder_last_hidden_states, encoder_hidden_states, encoder_attentions
        return (loss, logits) + outputs[1:]


class CPTForQuestionAnswering(nn.Module):
    def __init__(self, cls_mode=3, d_model: int = 1024, vocab_size: int = 50265,
                 num_encoder_layers: int = 12, num_decoder_layers: int = 12,
                 encoder_attn_heads: int = 16, decoder_attn_heads: int = 16,
                 encoder_ffn_dim: int = 4096, decoder_ffn_dim: int = 4096,
                 max_position_embeddings: int = 1024, activation="gelu",
                 pad_token_id: int = 1, attn_dropout: float = 0.0,
                 hidden_dropout: float = 0.0, act_dropout=0.0,
                 decoder_layerdrop: float = 0.0, scale_embedding: bool = False,
                 decoder_start_token_id=2):
        super(CPTForQuestionAnswering, self).__init__()

        self.num_labels = 2

        self.model = CPT(d_model, vocab_size, num_encoder_layers, num_decoder_layers, encoder_attn_heads, decoder_attn_heads, encoder_ffn_dim, decoder_ffn_dim, max_position_embeddings,
                          activation, pad_token_id, attn_dropout, hidden_dropout, act_dropout, decoder_layerdrop, scale_embedding, decoder_start_token_id)

        # Encoder for classification.
        if cls_mode == 1:
            cls_dim = d_model
        # Decoder for classification.
        elif cls_mode == 2:
            cls_dim = d_model
        # Both encoder & decoder for classification.'
        elif cls_mode == 3:
            cls_dim = d_model * 2
        else:
            raise NotImplementedError
        
        self.qa_outputs = nn.Linear(cls_dim, self.num_labels)

        init_weights(self.qa_outputs)
        self.cls_mode = cls_mode

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        encoder_outputs=None,
        start_positions=None,
        end_positions=None,
        inputs_embeds=None,
        decoder_inputs_embeds=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
    ):
        r"""
        start_positions (:obj:`flow.Tensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for position (index) of the start of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
            are not taken into account for computing the loss.
        end_positions (:obj:`flow.Tensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for position (index) of the end of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
            are not taken into account for computing the loss.
        """
        if start_positions is not None and end_positions is not None:
            use_cache = False

        if input_ids is None and inputs_embeds is not None:
            raise NotImplementedError(
                f"Passing input embeddings is currently not supported for {self.__class__.__name__}"
            )

        outputs = self.model(input_ids, attention_mask, decoder_input_ids, decoder_attention_mask, head_mask, decoder_head_mask,
                             encoder_outputs, None, inputs_embeds, decoder_inputs_embeds, use_cache, output_attentions, output_hidden_states)

        hidden_states = outputs[0]
        enc_hidden_states = outputs[0]

        if self.cls_mode == 1:
            logits = self.qa_outputs(enc_hidden_states)
        elif self.cls_mode == 2:
            logits = self.qa_outputs(hidden_states)
        elif self.cls_mode == 3:
            rep = flow.cat([enc_hidden_states, hidden_states], dim=-1)
            logits = self.qa_outputs(rep)
        else:
            raise NotImplementedError

        # start_logits, end_logits = logits.split(1, dim=-1)
        # oneflow does not support split.
        split_half = logits.shape[-1] // 2
        start_logits, end_logits = logits[:, :,
                                          :split_half], logits[:, :, split_half:]
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        total_loss = None
        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions = start_positions.clamp(0, ignored_index)
            start_positions = end_positions.clamp(0, ignored_index)

            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2

        # total_loss, start_logits, end_logits, past_key_values,
        # decoder_hidden_states, decoder_attentions, cross_attentions,
        # encoder_last_hidden_states, encoder_hidden_states, encoder_attentions
        return (total_loss, start_logits, end_logits) + outputs[1:]