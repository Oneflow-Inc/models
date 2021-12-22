import oneflow as flow
import torch
import transformers
import models.roberta as roberta_flow
from base_weight_utils import(
    enter,
    quit,
    indent_msg,
    Linear_trans,
    Dropout_trans,
    LayerNorm_trans,
    Embedding_trans    
)


def RobertaEmbeddings_trans(model_flow, model_torch):
    print(" RobertaEmbedding")
    enter()
    assert isinstance(model_flow, roberta_flow.RobertaEmbeddings)

    indent_msg("word_embedding:")
    model_flow.word_embeddings = Embedding_trans(
        model_flow.word_embeddings, model_torch.word_embeddings)
    indent_msg("position_embedding:")
    model_flow.position_embeddings = Embedding_trans(
        model_flow.position_embeddings, model_torch.position_embeddings)
    indent_msg("token_type_embeddings:")
    model_flow.token_type_embeddings = Embedding_trans(
        model_flow.token_type_embeddings, model_torch.token_type_embeddings)

    indent_msg("LayerNorm:")
    model_flow.LayerNorm = LayerNorm_trans(
        model_flow.LayerNorm, model_torch.LayerNorm)
    indent_msg("dropout:")
    model_flow.dropout = Dropout_trans(model_flow.dropout, model_torch.dropout)

    # # buffer
    # model_flow.position_embedding_type = model_torch.position_embedding_type
    # model_flow.token_type_ids = model_torch.token_type_ids
    model_flow.padding_idx = model_torch.padding_idx

    quit()
    return model_flow


def RobertaSelfAttention_trans(model_flow, model_torch):
    print(" RobertaSelfAttention")
    enter()
    assert isinstance(model_flow, roberta_flow.RobertaSelfAttention)

    model_flow.num_attention_heads = model_torch.num_attention_heads
    model_flow.attention_head_size = model_torch.attention_head_size
    model_flow.all_head_size = model_torch.all_head_size

    indent_msg("query:")
    model_flow.query = Linear_trans(model_flow.query, model_torch.query)
    indent_msg("key:")
    model_flow.key = Linear_trans(model_flow.key, model_torch.key)
    indent_msg("value:")
    model_flow.value = Linear_trans(model_flow.value, model_torch.value)

    indent_msg("dropout:")
    model_flow.dropout = Dropout_trans(model_flow.dropout, model_torch.dropout)
    model_flow.position_embedding_type = model_torch.position_embedding_type
    if hasattr(model_flow, "max_position_embeddings") and hasattr(model_torch, "max_positoin_embeddings"):
        model_flow.max_position_embeddings = model_torch.max_position_embeddings
    elif hasattr(model_flow, "max_position_embeddings") or hasattr(model_torch, "max_positoin_embeddings"):
        print("max_position_embeddings: mode_flow:", hasattr(model_flow, "max_position_embeddings"),
              ", model_torch", hasattr(model_torch, "max_positoin_embeddings"))

    if hasattr(model_flow, "distance_embedding") and hasattr(model_torch, "distance_embedding"):
        indent_msg("distance_embedding:")
        model_flow.distance_embedding = Embedding_trans(
            model_flow.distance_embedding, model_torch.distance_embedding)
    elif hasattr(model_flow, "distance_embedding") or hasattr(model_torch, "distance_embedding"):
        print("distance_embedding: mode_flow:", hasattr(model_flow, "distance_embedding"),
              ", model_torch:", hasattr(model_torch, "distance_embedding"))

    model_flow.is_decoder = model_torch.is_decoder

    quit()
    return model_flow


def RobertaSelfOutput_trans(model_flow, model_torch):
    print(" RobertaSelfOutput")
    enter()
    assert isinstance(model_flow, roberta_flow.RobertaSelfOutput)

    indent_msg("dense:")
    model_flow.dense = Linear_trans(model_flow.dense, model_torch.dense)
    indent_msg("LayerNorm:")
    model_flow.LayerNorm = LayerNorm_trans(
        model_flow.LayerNorm, model_torch.LayerNorm)
    indent_msg("dropout")
    model_flow.dropout = Dropout_trans(model_flow.dropout, model_torch.dropout)

    quit()
    return model_flow


def RobertaAttention_trans(model_flow, model_torch):
    print(" RobertaAttention")
    enter()
    assert isinstance(model_flow, roberta_flow.RobertaAttention)

    indent_msg("selfattn:")
    model_flow.selfattn = RobertaSelfAttention_trans(
        model_flow.selfattn, model_torch.self)
    indent_msg("output:")
    model_flow.output = RobertaSelfOutput_trans(
        model_flow.output, model_torch.output)
    model_flow.pruned_heads = model_torch.pruned_heads

    quit()
    return model_flow


def RobertaIntermediate_trans(model_flow, model_torch):
    print(" RobertaIntermediate")
    enter()
    assert isinstance(model_flow, roberta_flow.RobertaIntermediate)

    indent_msg("dense:")
    model_flow.dense = Linear_trans(model_flow.dense, model_torch.dense)
    # intermediate_act_fn

    quit()
    return model_flow


def RobertaOutput_trans(model_flow, model_torch):
    print(" RobertaOutput")
    enter()
    assert isinstance(model_flow, roberta_flow.RobertaOutput)

    indent_msg("dense:")
    model_flow.dense = Linear_trans(model_flow.dense, model_torch.dense)
    indent_msg("LayerNorm")
    model_flow.LayerNorm = LayerNorm_trans(
        model_flow.LayerNorm, model_torch.LayerNorm)
    indent_msg("dropout:")
    model_flow.dropout = Dropout_trans(model_flow.dropout, model_torch.dropout)

    quit()
    return model_flow


def RobertaLayer_trans(model_flow, model_torch):
    print(" RobertaLayer")
    enter()
    assert isinstance(model_flow, roberta_flow.RobertaLayer)

    model_flow.chunk_size_feed_forward = model_torch.chunk_size_feed_forward
    indent_msg("attention:")
    model_flow.attention = RobertaAttention_trans(
        model_flow.attention, model_torch.attention)
    model_flow.is_decoder = model_torch.is_decoder
    model_flow.add_cross_attention = model_torch.add_cross_attention

    if hasattr(model_flow, "crossattention") and hasattr(model_torch, "crossattention"):
        indent_msg("crossattention")
        model_flow.crossattention = RobertaAttention_trans(
            model_flow.crossattention, model_torch.crossattention)
    elif hasattr(model_flow, "crossattention") or hasattr(model_torch, "crossattention"):
        print("crossattention: mode_flow:", hasattr(model_flow, "crossattention"),
              ", model_torch:", hasattr(model_torch, "crossattention"))

    indent_msg("intermediate:")
    model_flow.intermediate = RobertaIntermediate_trans(
        model_flow.intermediate, model_torch.intermediate)
    indent_msg("output:")
    model_flow.output = RobertaOutput_trans(
        model_flow.output, model_torch.output)

    quit()
    return model_flow


def RobertaEncoder_trans(model_flow, model_torch):
    print(" RobertaEncoder")
    enter()
    assert isinstance(model_flow, roberta_flow.RobertaEncoder)

    model_flow.add_cross_attention = model_torch.config.add_cross_attention

    assert model_flow.num_layers == model_torch.config.num_hidden_layers
    for i in range(model_flow.num_layers):
        indent_msg("layer {}:".format(i))
        model_flow.layer[i] = RobertaLayer_trans(
            model_flow.layer[i], model_torch.layer[i])

    quit()
    return model_flow


def RobertaPooler_trans(model_flow, model_torch):
    print(" RobertaPooler")
    enter()
    assert isinstance(model_flow, roberta_flow.RobertaPooler)

    indent_msg("dense:")
    model_flow.dense = Linear_trans(model_flow.dense, model_torch.dense)

    quit()
    return model_flow


def Roberta_trans(model_flow, model_torch):
    print(" Roberta")
    enter()
    assert isinstance(model_flow, roberta_flow.Roberta)
    assert isinstance(model_torch, transformers.RobertaModel)

    indent_msg("embeddings:")
    model_flow.embeddings = RobertaEmbeddings_trans(
        model_flow.embeddings, model_torch.embeddings)
    indent_msg("encoder:")
    model_flow.encoder = RobertaEncoder_trans(
        model_flow.encoder, model_torch.encoder)
    if model_torch.pooler == None:
        assert model_flow.pooler == None
    if model_torch.pooler != None:
        assert model_flow.pooler != None
        indent_msg("pooler:")
        model_flow.pooler = RobertaPooler_trans(
            model_flow.pooler, model_torch.pooler)

    quit()
    return model_flow

def RobertaLMHead_trans(model_flow, model_torch):
    print(" RobertaLMHead")
    enter()

    assert isinstance(model_flow, roberta_flow.RobertaLMHead)

    indent_msg("dense:")
    model_flow.dense = Linear_trans(model_flow.dense, model_torch.dense)
    indent_msg("layer_norm:")
    model_flow.layer_norm = LayerNorm_trans(model_flow.layer_norm, model_torch.layer_norm)
    indent_msg("decoder:")
    model_flow.decoder= Linear_trans(model_flow.decoder, model_torch.decoder)

    quit()
    return model_flow

def RobertaClassificationHead_trans(model_flow, model_torch):
    print("  RobertaClassificationHead")
    enter()

    assert isinstance(model_flow, roberta_flow.RobertaClassificationHead)

    indent_msg("dense:")
    model_flow.dense = Linear_trans(model_flow.dense, model_torch.dense)
    indent_msg("dropout:")
    model_flow.dropout = Dropout_trans(model_flow.dropout, model_torch.dropout)
    indent_msg("out_proj")
    model_flow.out_proj = Linear_trans(model_flow.out_proj, model_torch.out_proj)

    quit()
    return model_flow

def RobertaForCausalLM_trans(model_flow, model_torch):
    print(" RobertaForCausalLM")
    enter()

    assert isinstance(model_flow, roberta_flow.RobertaForCausalLM)
    assert isinstance(model_torch, transformers.RobertaForCausalLM)

    assert model_flow.vocab_size == model_torch.config.vocab_size
    indent_msg("roberta:")
    model_flow.roberta = Roberta_trans(model_flow.roberta, model_torch.roberta)
    indent_msg("lm_head:")
    model_flow.lm_head == RobertaLMHead_trans(model_flow.lm_head, model_torch.lm_head)

    quit()
    return model_flow

def RobertaForMaskedLM_trans(model_flow, model_torch):
    print(" RobertaForMaskedLM")
    enter()

    assert isinstance(model_flow, roberta_flow.RobertaForMaskedLM)
    assert isinstance(model_torch, transformers.RobertaForMaskedLM)

    assert model_flow.vocab_size == model_torch.config.vocab_size
    indent_msg("roberta:")
    model_flow.roberta = Roberta_trans(model_flow.roberta, model_torch.roberta)
    indent_msg("lm_head:")
    model_flow.lm_head == RobertaLMHead_trans(model_flow.lm_head, model_torch.lm_head)

    quit()
    return model_flow

def RobertaForSequenceClassification_trans(model_flow, model_torch):
    print(" RobertaForSequenceClassification")
    enter()

    assert isinstance(model_flow, roberta_flow.RobertaForSequenceClassification)
    assert isinstance(model_torch, transformers.RobertaForSequenceClassification)

    assert model_flow.problem_type == model_torch.config.problem_type
    assert model_flow.num_labels == model_torch.num_labels
    indent_msg("roberta:")
    model_flow.roberta = Roberta_trans(model_flow.roberta, model_torch.roberta)
    indent_msg("classifier:")
    model_flow.classifier = RobertaClassificationHead_trans(model_flow.classifier, model_torch.classifier)


    quit()
    return model_flow

def RobertaForMultipleChoice_trans(model_flow, model_torch):
    print(" RobertaForMultipleChoice")
    enter()

    assert isinstance(model_flow, roberta_flow.RobertaForMultipleChoice)
    assert isinstance(model_torch, transformers.RobertaForMultipleChoice)

    indent_msg("roberta:")
    model_flow.roberta = Roberta_trans(model_flow.roberta, model_torch.roberta)
    indent_msg("dropout:")
    model_flow.dropout = Dropout_trans(model_flow.dropout, model_torch.dropout)
    indent_msg("classifier")
    model_flow.classifier = Linear_trans(model_flow.classifier, model_torch.classifier)

    quit()
    return model_flow

def RobertaForTokenClassification_trans(model_flow, model_torch):
    print(" RobertaForTokenClassification")
    enter()

    assert isinstance(model_flow, roberta_flow.RobertaForTokenClassification)
    assert isinstance(model_torch, transformers.RobertaForTokenClassification)

    assert model_flow.num_labels == model_torch.num_labels
    indent_msg("roberta:")
    model_flow.roberta = Roberta_trans(model_flow.roberta, model_torch.roberta)
    indent_msg("dropout:")
    model_flow.dropout = Dropout_trans(model_flow.dropout, model_torch.dropout)
    indent_msg("classifier")
    model_flow.classifier = Linear_trans(model_flow.classifier, model_torch.classifier)

    quit()
    return model_flow

def RobertaForQuestionAnswering_trans(model_flow, model_torch):
    print(" RobertaForQuestionAnswering")
    enter()

    assert isinstance(model_flow, roberta_flow.RobertaForQuestionAnswering)
    assert isinstance(model_torch, transformers.RobertaForQuestionAnswering)

    assert model_flow.num_labels == model_torch.num_labels
    indent_msg("roberta:")
    model_flow.roberta = Roberta_trans(model_flow.roberta, model_torch.roberta)
    indent_msg("qa_outputs:")
    model_flow.qa_outputs = Linear_trans(model_flow.qa_outputs, model_torch.qa_outputs)

    quit()
    return model_flow