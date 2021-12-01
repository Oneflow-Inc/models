import oneflow as flow
from oneflow import nn
import math


class BertEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings.
    """

    def __init__(
        self,
        vocab_size,
        type_vocab_size,
        max_position_embeddings,
        hidden_size,
        hidden_dropout_prob,
        seq_length,
    ):
        super().__init__()
        self.word_embeddings = nn.Parameter(flow.empty((vocab_size, hidden_size), dtype=flow.float32))
        self.position_embeddings = nn.Parameter(flow.empty((max_position_embeddings, hidden_size), dtype=flow.float32))
        self.tokentype_embeddings = nn.Parameter(flow.empty((type_vocab_size, hidden_size), dtype=flow.float32))
        
        flow.nn.init.normal_(self.word_embeddings)
        flow.nn.init.normal_(self.position_embeddings)
        flow.nn.init.normal_(self.tokentype_embeddings)

        self.LayerNorm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(hidden_dropout_prob, inplace=True)
        self.register_buffer(
            "position_ids", flow.arange(seq_length).unsqueeze(0)
        )
        self.seq_length = seq_length

    def forward(self, input_ids, token_type_ids, position_ids=None):

        if False:
            word_embeddings = flow._C.amp_white_identity(self.word_embeddings)
            position_embeddings = flow._C.amp_white_identity(self.position_embeddings)
            tokentype_embeddings = flow._C.amp_white_identity(self.tokentype_embeddings)

        else:
            word_embeddings = self.word_embeddings
            position_embeddings = self.position_embeddings
            tokentype_embeddings = self.tokentype_embeddings

        input_embeds = flow._C.gather(word_embeddings, input_ids, axis=0)

        if position_ids is None:
            position_ids = self.position_ids
        
        position_embeds = flow._C.gather(position_embeddings, position_ids, axis=0)

        token_type_embeds = flow._C.gather(tokentype_embeddings, token_type_ids, axis=0)

        embeddings = input_embeds + token_type_embeds + position_embeds
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class BertSelfAttention(nn.Module):
    def __init__(
        self,
        num_attention_heads,
        hidden_size,
        seq_len,
        attention_probs_dropout_prob=0.0,
    ):
        super().__init__()
        assert hidden_size % num_attention_heads == 0

        self.num_attention_heads = num_attention_heads
        self.attention_head_size = int(hidden_size / num_attention_heads)
        self.all_head_size = num_attention_heads * self.attention_head_size
        self.seq_len = seq_len

        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(attention_probs_dropout_prob, inplace=True)

    def transpose_for_scores(self, x):
        x = flow.reshape(
            x, [-1, self.seq_len, self.num_attention_heads, self.attention_head_size]
        )
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask):
        # hidden_states: [batch_size * seq, hidden_size]
        
        query_layer = self.transpose_for_scores(self.query(hidden_states))
        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))

        # attention_scores = flow.matmul(query_layer, key_layer.transpose(-2, -1))
        attention_scores = flow.matmul(query_layer, key_layer, transpose_b=True)
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = flow.softmax(attention_scores, dim=-1)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        context_layer = flow.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3)

        context_layer = flow.reshape(
            context_layer, [-1, self.all_head_size]
        )
        return context_layer


class BertSelfOutput(nn.Module):
    def __init__(self, hidden_size, hidden_dropout_prob):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.LayerNorm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(hidden_dropout_prob, inplace=True)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertAttention(nn.Module):
    def __init__(
        self,
        num_attention_heads,
        hidden_size,
        seq_len,
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
    ):
        super().__init__()
        self.self = BertSelfAttention(
            num_attention_heads, hidden_size, seq_len, attention_probs_dropout_prob
        )
        self.output = BertSelfOutput(hidden_size, hidden_dropout_prob)

    def forward(self, hidden_states, attention_mask):
        self_attention_output = self.self(hidden_states, attention_mask)
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
        self.LayerNorm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(hidden_dropout_prob, inplace=True)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertLayer(nn.Module):
    def __init__(
        self,
        num_attention_heads,
        hidden_size,
        seq_len,
        intermediate_size,
        hidden_act,
        hidden_dropout_prob=0.1,
        attention_prob_dropout_prob=0.1,
    ):
        super().__init__()
        self.attention = BertAttention(
            num_attention_heads,
            hidden_size,
            seq_len,
            hidden_dropout_prob,
            attention_prob_dropout_prob,
        )

        self.intermediate = BertIntermediate(hidden_size, intermediate_size, hidden_act)
        self.output = BertOutput(intermediate_size, hidden_size, hidden_dropout_prob)

    def forward(self, hidden_states, attention_mask):
        self_attention_output = self.attention(hidden_states, attention_mask)
        intermediate_output = self.intermediate(self_attention_output)
        layer_output = self.output(intermediate_output, self_attention_output)
        return layer_output


class BertEncoder(nn.Module):
    def __init__(
        self,
        num_hidden_layers,
        num_attention_heads,
        hidden_size,
        seq_len,
        intermediate_size,
        hidden_act,
        hidden_dropout_prob,
        attention_prob_dropout_prob,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.seq_len = seq_len

        self.layer = nn.ModuleList(
            [
                BertLayer(
                    num_attention_heads,
                    hidden_size,
                    seq_len,
                    intermediate_size,
                    hidden_act,
                    hidden_dropout_prob,
                    attention_prob_dropout_prob,
                )
                for _ in range(num_hidden_layers)
            ]
        )

    def forward(self, hidden_states, attention_mask):
        hidden_states = hidden_states.reshape([-1, self.hidden_size])
        for layer_module in self.layer:
            hidden_states = layer_module(hidden_states, attention_mask)
        hidden_states = hidden_states.reshape([-1, self.seq_len, self.hidden_size]) 
        return hidden_states


class BertPooler(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size

        self.dense = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        """Just "pool" the model by simply taking the [CLS] token corresponding to the first token.
        """
        first_token_tensor = hidden_states[:, 0]
        first_token_tensor = flow.reshape(first_token_tensor, [-1, self.hidden_size])
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class BertModel(nn.Module):
    def __init__(
        self,
        vocab_size,
        seq_length,
        hidden_size,
        num_hidden_layers,
        num_attention_heads,
        intermediate_size,
        hidden_act=nn.GELU(),
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=512,
        type_vocab_size=16,
    ):
        super().__init__()

        self.embeddings = BertEmbeddings(
            vocab_size,
            type_vocab_size,
            max_position_embeddings,
            hidden_size,
            hidden_dropout_prob,
            seq_length,
        )
        self.encoder = BertEncoder(
            num_hidden_layers,
            num_attention_heads,
            hidden_size,
            seq_length,
            intermediate_size,
            hidden_act,
            hidden_dropout_prob,
            attention_probs_dropout_prob,
        )

        self.pooler = BertPooler(hidden_size)

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def forward(self, input_ids, token_type_ids, attention_mask):
        input_shape = input_ids.size()
        seq_length = input_shape[1]

        embedding_outputs = self.embeddings(input_ids, token_type_ids)

        extended_attention_mask = self.get_extended_attention_mask(
            attention_mask, seq_length, seq_length
        )

        encoder_outputs = self.encoder(embedding_outputs, extended_attention_mask)

        sequence_output = encoder_outputs
        pooled_output = self.pooler(sequence_output)
        return sequence_output, pooled_output

    def get_extended_attention_mask(
        self, attention_mask, from_seq_length, to_seq_length
    ):
        output = flow.cast(attention_mask, dtype=flow.float32)
        output = flow.reshape(output, [-1, 1, to_seq_length])
        # broadcast `from_tensor` from 2D to 3D
        output = output.expand(-1, from_seq_length, -1)

        attention_mask = flow.reshape(output, [-1, 1, from_seq_length, to_seq_length])
        attention_mask = flow.cast(attention_mask, dtype=flow.float32)
        addr_blob = (attention_mask - 1.0) * 10000.0
        return addr_blob


class BertPredictionHeadTransform(nn.Module):
    def __init__(self, hidden_size, hidden_act=nn.GELU()):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.transform_act_fn = hidden_act
        self.LayerNorm = nn.LayerNorm(hidden_size)

    def forward(self, sequence_output):
        sequence_output = self.dense(sequence_output)
        sequence_output = self.transform_act_fn(sequence_output)
        sequence_output = self.LayerNorm(sequence_output)
        return sequence_output


class BertLMPredictionHead(nn.Module):
    def __init__(self, hidden_size, vocab_size, hidden_act=nn.GELU()):
        super().__init__()
        self.hidden_size = hidden_size

        self.transform = BertPredictionHeadTransform(hidden_size, hidden_act)

        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        self.decoder = nn.Linear(hidden_size, vocab_size, bias=False)

        self.output_bias = nn.Parameter(flow.zeros(vocab_size))

        # Need a link between the two variables so that the bias is correctly resized with `resize_token_embeddings`
        self.decoder.bias = self.output_bias

    def forward(self, sequence_output, masked_lm_positions):
        # Gather masked outputs
        masked_sequence_output = flow.gather(sequence_output, index=masked_lm_positions.unsqueeze(2).expand(-1, -1, self.hidden_size), dim=1)
        masked_sequence_output = masked_sequence_output.reshape([-1, self.hidden_size])

        masked_sequence_output = self.transform(masked_sequence_output)
        masked_sequence_output = self.decoder(masked_sequence_output)
        return masked_sequence_output


class BertPreTrainingHeads(nn.Module):
    def __init__(self, hidden_size, vocab_size, hidden_act=nn.GELU()):
        super().__init__()
        self.predictions = BertLMPredictionHead(hidden_size, vocab_size, hidden_act)
        self.seq_relationship = nn.Linear(hidden_size, 2)

    def forward(self, sequence_output, pooled_output, masked_lm_positions):
        prediction_scores = self.predictions(sequence_output, masked_lm_positions)
        seq_relationship_scores = self.seq_relationship(pooled_output)
        return prediction_scores, seq_relationship_scores


class BertForPreTraining(nn.Module):
    def __init__(
        self,
        vocab_size,
        seq_length,
        hidden_size,
        hidden_layers,
        atten_heads,
        intermediate_size,
        hidden_act,
        hidden_dropout_prob,
        attention_probs_dropout_prob,
        max_position_embeddings,
        type_vocab_size,
        initializer_range=0.02,
    ):
        super().__init__()
        self.initializer_range = initializer_range

        self.bert = BertModel(
            vocab_size,
            seq_length,
            hidden_size,
            hidden_layers,
            atten_heads,
            intermediate_size,
            hidden_act,
            hidden_dropout_prob,
            attention_probs_dropout_prob,
            max_position_embeddings,
            type_vocab_size,
        )

        self.cls = BertPreTrainingHeads(hidden_size, vocab_size)

        self.init_weights()

    def forward(self, input_ids, token_type_ids, attention_mask, masked_lm_positions):
        sequence_output, pooled_output = self.bert(
            input_ids, token_type_ids, attention_mask
        )

        prediction_scores, seq_relationship_scores = self.cls(
            sequence_output, pooled_output, masked_lm_positions
        )
        return prediction_scores, seq_relationship_scores

    def get_output_embeddings(self):
        return self.cls.predictions.decoder

    def set_output_embeddings(self, new_embeddings):
        self.cls.predictions.decoder = new_embeddings

    def init_weights(self):
        self.apply(self._init_weights)

        self.clone_weights(
            self.get_output_embeddings(), self.bert.get_input_embeddings()
        )

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, nn.Linear):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
            if module.bias is not None:
                module.bias.data.fill_(0.0)
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.weight.data.fill_(1.0)
            module.bias.data.fill_(0.0)

    def clone_weights(self, output_embeddings, input_embeddings):
        """
        Tie the weights between the input embeddings and the output embeddings.
        """
        output_embeddings.weight = input_embeddings
