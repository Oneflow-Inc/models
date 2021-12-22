import oneflow as flow
from models.roberta import (
    Roberta, 
    RobertaForCausalLM, 
    RobertaForMaskedLM, 
    RobertaForMultipleChoice,
    RobertaForSequenceClassification,
    RobertaForTokenClassification,
    RobertaForQuestionAnswering,
)

def get_random_tensor(low, high, sz):
        return flow.Tensor(np.random.randint(low=low, size=sz, high=high)).to("cuda")

if __name__ == '__main__':

    import numpy as np
    # init parameters
    # see https://huggingface.co/transformers/model_doc/bert.html#bertconfig
    # and https://huggingface.co/transformers/_modules/transformers/models/roberta/configuration_roberta.html#RobertaConfig
    # for details.
    bs = 16
    seq_len = 128
    vocab_size = 2000
    type_vocab_size = 2
    max_position_embeddings = 512
    hidden_size = 768
    intermediate_size = 2048
    chunk_size_feed_forward = 0
    num_layers = 6
    nheads = 6
    activation = "relu"
    pad_token_id = 1  # code the pad word.
    layer_norm_eps = 1e-5
    attn_dropout = 0.1
    hidden_dropout = 0.1
    position_embedding_type = "absolute"
    is_decoder = False
    add_pooling_layer = True
    add_cross_attention = False

    input_ids = get_random_tensor(0, 2000, (bs, seq_len)).to(flow.int32)
    attention_mask = get_random_tensor(0, 2, sz=(bs, seq_len))
    attention_mask = None
    token_type_ids = get_random_tensor(0, 2, (bs, seq_len)).to(flow.int32)
    #token_type_ids = None
    position_ids = get_random_tensor(
        1, max_position_embeddings - 1, (bs, seq_len)).to(flow.int32)
    #position_ids = None
    head_mask = get_random_tensor(0, 2, (num_layers, nheads))
    head_mask = None
    inputs_embeds = None
    output_attentions = True
    output_hidden_states = True
    encoder_hidden_states = get_random_tensor(0, 5, (bs, seq_len, hidden_size))
    encoder_hidden_states = None
    encoder_attention_mask = get_random_tensor(0, 5, (bs, seq_len))
    encoder_attention_mask = None
    past_key_values = tuple([
        tuple([
            get_random_tensor(
                0, 2, (bs, nheads, seq_len, seq_len))
            for i in range(4)
        ])
        for j in range(num_layers)
    ])
    use_cache = False
    past_key_values = None

    print("Testing roberta...", end="")

    model = Roberta(vocab_size, type_vocab_size, max_position_embeddings, hidden_size, intermediate_size, chunk_size_feed_forward, num_layers, nheads,
                    activation, pad_token_id, layer_norm_eps, attn_dropout, hidden_dropout, position_embedding_type, is_decoder, add_pooling_layer, add_cross_attention).to("cuda")
    output = model(
        input_ids,
        attention_mask,
        token_type_ids,
        position_ids,
        head_mask,
        inputs_embeds,
        encoder_hidden_states,
        encoder_attention_mask,
        past_key_values,
        use_cache,
        output_attentions,
        output_hidden_states
    )
    print("Done.")

    print("Testing RobertaForCausalLM...", end="end")
    labels = get_random_tensor(-100, vocab_size, (bs, seq_len)).to(flow.int64)
    model = RobertaForCausalLM(vocab_size, type_vocab_size, max_position_embeddings, hidden_size, intermediate_size, chunk_size_feed_forward, num_layers, nheads,
                    activation, pad_token_id, layer_norm_eps, attn_dropout, hidden_dropout, position_embedding_type, is_decoder, add_cross_attention).to("cuda")
    output = model(
        input_ids,
        attention_mask,
        token_type_ids,
        position_ids,
        head_mask,
        inputs_embeds,
        encoder_hidden_states,
        encoder_attention_mask,
        labels,
        past_key_values,
        use_cache,
        output_attentions,
        output_hidden_states
    )
    print("Done.")

    print("Testing RobertaForMaskedLM...", end="")
    model = RobertaForMaskedLM(vocab_size, type_vocab_size, max_position_embeddings, hidden_size, intermediate_size, chunk_size_feed_forward, num_layers, nheads,
                    activation, pad_token_id, layer_norm_eps, attn_dropout, hidden_dropout, position_embedding_type, is_decoder, add_cross_attention).to("cuda")
    output = model(
        input_ids,
        attention_mask,
        token_type_ids,
        position_ids,
        head_mask,
        inputs_embeds,
        encoder_hidden_states,
        encoder_attention_mask,
        labels,
        output_attentions,
        output_hidden_states
    )
    print("Done.")

    num_labels = 2
    problem_type = None
    print("Testing RobertaForSequenceClassification...", end="")
    labels = get_random_tensor(0, num_labels, (bs,)).to(flow.int64)
    model = RobertaForSequenceClassification(num_labels, vocab_size, type_vocab_size, max_position_embeddings, hidden_size, intermediate_size, chunk_size_feed_forward, num_layers, nheads,
                    activation, pad_token_id, layer_norm_eps, attn_dropout, hidden_dropout, position_embedding_type, is_decoder, add_cross_attention).to("cuda")
    output = model(
        input_ids,
        attention_mask,
        token_type_ids,
        position_ids,
        head_mask,
        inputs_embeds,
        labels,
        output_attentions,
        output_hidden_states
    )
    print("Done.")

    print("Testing RobertaForMultipleChoice...", end="")
    num_choices = 5
    input_choices = get_random_tensor(0, 2000, (bs, num_choices, seq_len)).to(flow.int32)
    attention_mask_choices = get_random_tensor(0, 2, sz=(bs, num_choices, seq_len))
    attention_mask_choices = None
    token_type_ids_choices = get_random_tensor(0, 2, (bs, num_choices, seq_len)).to(flow.int32)
    position_ids_choices = get_random_tensor(
        1, max_position_embeddings - 1, (bs, num_choices, seq_len)).to(flow.int32)
    model = RobertaForMultipleChoice(vocab_size, type_vocab_size, max_position_embeddings, hidden_size, intermediate_size, chunk_size_feed_forward, num_layers, nheads,
                    activation, pad_token_id, layer_norm_eps, attn_dropout, hidden_dropout, position_embedding_type, is_decoder, add_pooling_layer, add_cross_attention).to("cuda")
    output = model(
        input_choices,
        attention_mask_choices,
        token_type_ids_choices,
        labels,
        position_ids_choices,
        head_mask,
        inputs_embeds,
        output_attentions,
        output_hidden_states
    )
    print("Done.")

    print("Testing RobertaForTokenClassification...", end="")
    labels = get_random_tensor(0, num_labels, (bs,seq_len)).to(flow.int64)
    model = RobertaForTokenClassification(num_labels, vocab_size, type_vocab_size, max_position_embeddings, hidden_size, intermediate_size, chunk_size_feed_forward, num_layers, nheads,
                    activation, pad_token_id, layer_norm_eps, attn_dropout, hidden_dropout, position_embedding_type, is_decoder, add_cross_attention).to("cuda")
    output = model(
        input_ids,
        attention_mask,
        token_type_ids,
        position_ids,
        head_mask,
        inputs_embeds,
        labels,
        output_attentions,
        output_hidden_states
    )
    print("Done.")

    print("Testing RobertaForQuestionAnswering...", end="")
    start_pos = get_random_tensor(0, 40, (bs, )).to(flow.int32)
    end_pos = get_random_tensor(41, 60, (bs, )).to(flow.int32)
    model = RobertaForQuestionAnswering(num_labels, vocab_size, type_vocab_size, max_position_embeddings, hidden_size, intermediate_size, chunk_size_feed_forward, num_layers, nheads,
                    activation, pad_token_id, layer_norm_eps, attn_dropout, hidden_dropout, position_embedding_type, is_decoder, add_cross_attention).to("cuda")
    output = model(
        input_ids,
        attention_mask,
        token_type_ids,
        position_ids,
        head_mask,
        inputs_embeds,
        start_pos, 
        end_pos,
        output_attentions,
        output_hidden_states
    )
    print("Done.")