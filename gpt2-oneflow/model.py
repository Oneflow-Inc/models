
import math
import numpy as np
import oneflow as flow
import oneflow.nn as nn

def gelu(x):
    """
    Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT). Also see
    the Gaussian Error Linear Units paper: https://arxiv.org/abs/1606.08415
    """
    return 0.5 * x * (1.0 + flow.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * flow.pow(x, 3.0))))

class LayerNorm(nn.Module):
    "Construct a layernorm module."

    def __init__(self, hidden_size, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.eps = eps
        self.weight = nn.Parameter(flow.ones(hidden_size, dtype=flow.float32))
        self.bias = nn.Parameter(flow.zeros(hidden_size, dtype=flow.float32))

    def forward(self, x):
        # pytorch和tensorflow有细微差别，pytorch计算时eps在sqrt外面，tensorflow的eps在sqrt里面
        mean = x.mean(-1, keepdim=True)
        std = (x - mean).pow(2).mean(-1, keepdim=True)
        x = (x - mean) / flow.sqrt(std + self.eps)
        return self.weight * x + self.bias

class Conv1D(nn.Module):
    """
    1D-convolutional layer as defined by Radford et al. for OpenAI GPT (and also used in GPT-2).

    Basically works like a linear layer but the weights are transposed.

    Args:
        nf (:obj:`int`): The number of output features.
        nx (:obj:`int`): The number of input features.
    """
    def __init__(self, nf, nx):
        super(Conv1D, self).__init__()
        self.nf = nf
        self.weight = nn.Parameter(flow.Tensor(nx, nf))
        nn.init.normal_(self.weight, mean=0, std=0.02)
        self.bias = nn.Parameter(flow.zeros(nf))

    def forward(self, x):
        bsz, seq_len, channels = x.size()
        # size_out = x.size()[:-1] + (self.nf,)
        x = flow.addmm(self.bias, x.reshape((-1, channels)), self.weight)
        x = x.reshape([bsz, seq_len, self.nf])
        return x

class GPT2Attention(nn.Module):
    def __init__(self, config):
        super(GPT2Attention, self).__init__()

        max_positions = config.max_position_embeddings
        
        bias = np.tril(np.ones((max_positions, max_positions), dtype=np.uint8)).reshape((1, 1, max_positions, max_positions))
        self.register_buffer(
            "bias", flow.tensor(bias)
        )
        self.register_buffer("masked_bias", flow.tensor(-1e4).reshape((1, 1, 1, 1)))
        # self.bias = nn.Parameter(flow.tensor(bias))
        
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        assert self.embed_dim % self.num_heads == 0
        self.head_dim = self.embed_dim // self.num_heads
        self.scale_attn_weights = config.scale_attn_weights

        self.c_attn = Conv1D(self.embed_dim * 3, self.embed_dim)
        self.c_proj = Conv1D(self.embed_dim, self.embed_dim)

        self.attn_dropout = nn.Dropout(config.attn_pdrop)
        self.resid_dropout = nn.Dropout(config.resid_pdrop)

    def _attn(self, query, key, value):
        attn_weights = flow.matmul(query, key.transpose(-2, -1))

        if self.scale_attn_weights:
            attn_weights = attn_weights / (float(value.size(-1)) ** 0.5)

        query_length, key_length = query.size(-2), key.size(-2)
        causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]
        causal_mask = flow.broadcast_like(causal_mask.to(attn_weights.dtype), attn_weights).to(flow.int)    # broadcast_like会改变tensor类型
        masked_bias = flow.broadcast_like(self.masked_bias, attn_weights)
        attn_weights = flow.where(causal_mask, attn_weights, masked_bias)
        # attn_weights = attn_weights.masked_fill(causal_mask, 1e-4)    # value不支持tensor，bool和int混用，不支持广播

        attn_weights = nn.Softmax(dim=-1)(attn_weights)
        attn_weights = self.attn_dropout(attn_weights)

        attn_output = flow.matmul(attn_weights, value)
        return attn_output

    def _split_heads(self, tensor, num_heads, attn_head_size):
        """
        Splits hidden_size dim into attn_head_size and num_heads
        """
        bsz, seq_len = tensor.size()[:-1]
        new_shape = (bsz, seq_len, num_heads, attn_head_size)
        # new_shape = tensor.size()[:-1] + (num_heads, attn_head_size)
        tensor = tensor.reshape(new_shape)
        return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)

    def _merge_heads(self, tensor, num_heads, attn_head_size):
        """
        Merges attn_head_size dim and num_attn_heads dim into hidden_size
        """
        tensor = tensor.permute(0, 2, 1, 3)
        bsz, seq_len = tensor.size()[:-2]
        # new_shape = tensor.size()[:-2] + (num_heads * attn_head_size,)
        new_shape = (bsz, seq_len, num_heads * attn_head_size)
        return tensor.reshape(new_shape)

    def forward(self, hidden_states, layer_past=None, use_cache=False):
        hidden_states = self.c_attn(hidden_states)
        query, key, value = flow.chunk(hidden_states, chunks=3, dim=2)

        query = self._split_heads(query, self.num_heads, self.head_dim)
        key = self._split_heads(key, self.num_heads, self.head_dim)
        value = self._split_heads(value, self.num_heads, self.head_dim)

        if layer_past is not None:
            past_key, past_value = layer_past
            key = flow.cat((past_key, key), dim=-2)
            value = flow.cat((past_value, value), dim=-2)

        if use_cache is True:
            present = (key, value)
        else:
            present = None

        attn_output = self._attn(query, key, value)

        attn_output = self._merge_heads(attn_output, self.num_heads, self.head_dim)
        attn_output = self.c_proj(attn_output)
        attn_output = self.resid_dropout(attn_output)

        outputs = (attn_output, present)
        return outputs

class GPT2MLP(nn.Module):
    def __init__(self, intermediate_size, config):
        super().__init__()
        embed_dim = config.hidden_size
        self.c_fc = Conv1D(intermediate_size, embed_dim)
        self.c_proj = Conv1D(embed_dim, intermediate_size)
        self.act = gelu
        self.dropout = nn.Dropout(config.resid_pdrop)

    def forward(self, hidden_states):
        hidden_states = self.c_fc(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.c_proj(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states

class GPT2Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        hidden_size = config.hidden_size
        inner_dim = config.n_inner if config.n_inner is not None else 4 * hidden_size

        self.ln_1 = LayerNorm(hidden_size, eps=config.layer_norm_epsilon)
        self.attn = GPT2Attention(config)
        self.ln_2 = LayerNorm(hidden_size, eps=config.layer_norm_epsilon)
        self.mlp = GPT2MLP(inner_dim, config)
    
    def forward(self, hidden_states, layer_past=None, use_cache=False):
        residual = hidden_states
        hidden_states = self.ln_1(hidden_states)
        attn_output, present = self.attn(hidden_states, layer_past, use_cache)
        hidden_states = attn_output + residual

        residual = hidden_states
        hidden_states = self.ln_2(hidden_states)
        feed_forward_hidden_states = self.mlp(hidden_states)
        hidden_states = residual + feed_forward_hidden_states

        if use_cache:
            outputs = (hidden_states, present)
        else:
            outputs = (hidden_states,)
        return outputs
    
class GPT2Model(nn.Module):
    def __init__(self, config):
        super(GPT2Model, self).__init__()

        self.embed_dim = config.hidden_size

        self.wte = nn.Embedding(config.vocab_size, self.embed_dim)
        self.wpe = nn.Embedding(config.max_position_embeddings, self.embed_dim)
        
        self.drop = nn.Dropout(config.embd_pdrop)
        self.h = nn.ModuleList([GPT2Block(config) for _ in range(config.num_hidden_layers)])
        self.ln_f = LayerNorm(self.embed_dim, eps=config.layer_norm_epsilon)
    
    def forward(self, input_ids, position_ids=None, token_type_ids=None, past_key_values=None, use_cache=False):
        input_shape = input_ids.size()
        input_ids = input_ids.reshape((-1, input_ids.size(-1)))
        batch_size = input_ids.shape[0]

        if position_ids is not None:
            position_ids = position_ids.reshape((-1, input_shape[-1]))
        if token_type_ids is not None:
            token_type_ids = token_type_ids.reshape((-1, input_shape[-1]))
        
        if past_key_values is None:
            past_length = 0
            past_key_values = [None] * len(self.h)
        else:
            past_length = past_key_values[0][0].size(-2)

        if position_ids is None:
            position_ids = flow.arange(past_length, input_shape[-1] + past_length, dtype=flow.long, device=input_ids.device)
            position_ids = position_ids.unsqueeze(0).reshape((-1, input_shape[-1]))
        
        inputs_embeds = self.wte(input_ids)
        position_embeds = self.wpe(position_ids)
        hidden_states = inputs_embeds + position_embeds

        if token_type_ids is not None:
            token_type_embeds = self.wte(token_type_ids)
            hidden_states = hidden_states + token_type_embeds
        
        hidden_states = self.drop(hidden_states)

        presents = () if use_cache else None

        for i, (block, layer_past) in enumerate(zip(self.h, past_key_values)):
            outputs = block(hidden_states, layer_past, use_cache)
            hidden_states = outputs[0]
            if use_cache is True:
                presents = presents + (outputs[1],)
        
        hidden_states = self.ln_f(hidden_states)
        output_shape = (input_shape[0], input_shape[1], hidden_states.size(-1))

        hidden_states = hidden_states.reshape(output_shape)
        return tuple(v for v in [hidden_states, presents,] if v is not None)

class GPT2LMHeadModel(nn.Module):
    def __init__(self, config):
        super(GPT2LMHeadModel, self).__init__()
        self.transformer = GPT2Model(config)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # tie embeddings
        self.tie_embeddings()
    
    def tie_embeddings(self):
        self.lm_head.weight = self.transformer.wte.weight

    def forward(self, input_ids, position_ids=None, token_type_ids=None, labels=None, past_key_values=None, use_cache=False):
        hidden_states, past_key_values = self.transformer(input_ids, position_ids, token_type_ids, past_key_values, use_cache)
        lm_logits = self.lm_head(hidden_states)
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = lm_logits[..., :-1, :]
            shift_labels = labels[..., 1:]
            # Flatten the tokens
            loss_fct = nn.CrossEntropyLoss(ignore_index=-1)
            loss = loss_fct(shift_logits.reshape(-1, shift_logits.reshape(-1)), shift_labels.reshape(-1))
            return loss
        return lm_logits, past_key_values