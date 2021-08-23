import oneflow as flow
from oneflow import nn
import math


class BertModel(nn.Module):
    def __init__(self):
        super().__init__()


class BertEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings.
    """

    def __init__(
        self,
        vocab_size,
        type_vocab_size,
        max_position_embeddings,
        hidden_size,
        seq_length,
    ):
        super().__init__()
        self.word_embeddings = nn.Embedding(vocab_size, hidden_size)
        self.position_embeddings = nn.Embedding(max_position_embeddings, hidden_size)
        self.token_type_embeddings = nn.Embedding(type_vocab_size, hidden_size)

        self.layernorm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout()
        self.register_buffer(
            "position_ids", flow.arange(max_position_embeddings).unsqueeze(0)
        )
        self.seq_length = seq_length

    def forward(self, input_ids, token_type_ids, position_ids=None):

        input_embeds = self.word_embeddings(input_ids)

        if position_ids is None:
            position_ids = self.position_ids[:, : self.seq_length]
        position_embeds = self.position_embeddings(position_ids)

        token_type_embeds = self.token_type_embeddings(token_type_ids)

        embeddings = input_embeds + token_type_embeds + position_embeds
        embeddings = self.layernorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class BertSelfAttention(nn.Module):
    def __init__(self, num_attention_heads, hidden_size, seq_len, hidden_dropout_prob):
        super().__init__()

        self.num_attention_heads = num_attention_heads
        self.attention_head_size = int(hidden_size / num_attention_heads)
        self.all_head_size = num_attention_heads * self.attention_head_size
        self.seq_len = seq_len

        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(hidden_dropout_prob)

    def transpose_for_scores(self, x):
        x = flow.reshape(
            x, [-1, self.seq_len, self.num_attention_heads, self.attention_head_size]
        )
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask):
        query_layer = self.transpose_for_scores(self.query(hidden_states))
        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))

        attention_scores = flow.matmul(query_layer, key_layer.transpose(-2, -1))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = flow.softmax(attention_scores, dim=-1)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        context_layer = flow.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        context_layer = flow.reshape(
            context_layer, [-1, self.seq_len, self.all_head_size]
        )
        return context_layer


class BertSelfOutput(nn.Module):
    def __init__(self, hidden_size, hidden_dropout_prob):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.layernorm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.layernorm(hidden_states + input_tensor)
        return hidden_states


class BertAttention(nn.Module):
    def __init__(
        self, num_attention_heads, hidden_size, seq_len, hidden_dropout_prob=0.1
    ):
        super().__init__()
        self.self_atten = BertSelfAttention(
            num_attention_heads, hidden_size, seq_len, hidden_dropout_prob
        )
        self.output = BertSelfOutput(hidden_size, hidden_dropout_prob)

    def forward(self, hidden_states, attention_mask):
        self_attention_output = self.self_atten(hidden_states, attention_mask)
        output = self.output(self_attention_output, hidden_states)
        return output


class BertIntermediate(nn.Module):
    def __init__(self, hidden_size, intermediate_size, hidden_act=nn.GELU()):
        super().__init__()
        self.dense = nn.Linear(hidden_size, intermediate_size)
        self.intermediate_act_fn = hidden_act

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class BertOutput(nn.Module):
    def __init__(self, intermediate_size, hidden_size, hidden_dropout_prob=0.1):
        super().__init__()
        self.dense = nn.Linear(intermediate_size, hidden_size)
        self.layernorm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.layernorm(hidden_states + input_tensor)
        return hidden_states


class BertLayer(nn.Module):
    def __init__(
        self,
        num_attention_heads,
        hidden_size,
        seq_len,
        intermediate_size,
        hidden_dropout_prob=0.1,
    ):
        super().__init__()
        self.attention = BertAttention(
            num_attention_heads, hidden_size, seq_len, hidden_dropout_prob
        )

        self.intermediate = BertIntermediate(hidden_size, intermediate_size)
        self.output = BertOutput(intermediate_size, hidden_size, hidden_dropout_prob)

    def forward(self, hidden_states, attention_mask):
        self_attention_output = self.attention(hidden_states, attention_mask)
        intermediate_output = self.intermediate(self_attention_output)
        layer_output = self.output(intermediate_output, self_attention_output)
        return layer_output
