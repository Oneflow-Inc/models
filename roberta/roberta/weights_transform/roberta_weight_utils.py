import oneflow as flow
import torch
import transformers
import models.roberta as roberta_flow
from models.dev_ops import LayerNorm


def colored_string(string: str, color: str or int, end="\n") -> str:
    """output string in different color in cmd [This code is copied from fitlog]

    :param string: string to print
    :param color: color
    :return:
    """
    if isinstance(color, str):
        color = {
            "black": 30,
            "red": 31,
            "green": 32,
            "yellow": 33,
            "blue": 34,
            "purple": 35,
            "cyan": 36,
            "white": 37
        }[color]

    print("\033[%dm%s\033[0m" % (color, string), end=end)


DEPTH = 0


def indent_msg(msg, end=""):

    for i in range(DEPTH):
        if i == DEPTH - 1:
            print(" |-", end="")
        else:
            print(" | ", end="")
    colored_string(msg, color="yellow", end=end)


def enter():

    global DEPTH
    DEPTH += 1


def quit():

    global DEPTH
    DEPTH -= 1


def Parameter_trans(param_flow, param_torch):

    assert isinstance(param_flow, flow.nn.Parameter)
    assert isinstance(param_torch, torch.nn.Parameter)

    data_flow = param_flow.data
    data_torch = param_torch.data

    assert data_flow.dim() == data_torch.dim(
    ), "dimension not equal: flow {} vs torch {}.".format(data_flow.shape, data_torch.shape)
    for d_flow, d_torch in zip(data_flow.shape, data_torch.shape):
        assert d_flow == d_torch, "shapes not equal: flow {} vs torch {}.".format(
            data_flow.shape, data_torch.shape)

    if param_torch.device == "cpu":
        data = data_torch.detach().numpy()
    else:
        data = data_torch.cpu().detach().numpy()

    param_flow = flow.nn.Parameter(flow.tensor(data))

    return param_flow


def Embedding_trans(model_flow, model_torch):
    print(" Embedding")
    assert isinstance(model_flow, flow.nn.Embedding)
    assert isinstance(model_torch, torch.nn.Embedding)

    assert model_flow.num_embeddings == model_torch.num_embeddings, "num_embeddings not equal: flow {} vs torch {}.".format(
        model_flow.num_embeddings, model_torch.num_embeddings)
    assert model_flow.embedding_dim == model_torch.embedding_dim, "embedding_dim not equal: flow {} vs torch {}.".format(
        model_flow.embedding_dim, model_torch.embedding_dim)

    model_flow.padding_idx = model_torch.padding_idx
    model_flow.max_norm = model_torch.max_norm
    model_flow.norm_type = model_torch.norm_type
    model_flow.scale_grad_by_freq = model_torch.scale_grad_by_freq
    model_flow.sparse = model_torch.sparse

    model_flow.weight = Parameter_trans(model_flow.weight, model_torch.weight)

    return model_flow


def Linear_trans(model_flow, model_torch):
    print(" Linear")
    assert isinstance(model_flow, flow.nn.Linear)
    assert isinstance(model_torch, torch.nn.Linear)

    assert model_flow.in_features == model_torch.in_features, "in_features not equal: flow {} vs torch {}.".format(
        model_flow.in_features, model_torch.in_features)
    assert model_flow.out_features == model_torch.out_features, "out_features not equal: flow {} vs torch {}.".format(
        model_flow.out_features, model_torch.out_features)

    model_flow.weight = Parameter_trans(model_flow.weight, model_torch.weight)
    model_flow.bias = Parameter_trans(model_flow.bias, model_torch.bias)

    return model_flow


def LayerNorm_trans(model_flow, model_torch):
    print(" LayerNorm")
    assert isinstance(model_flow, LayerNorm)
    # assert isinstance(model_flow, flow.nn.LayerNorm)
    assert isinstance(model_torch, torch.nn.LayerNorm)

    model_flow.a_2 = Parameter_trans(model_flow.a_2, model_torch.weight)
    model_flow.b_2 = Parameter_trans(model_flow.b_2, model_torch.bias)
    model_flow.eps = model_torch.eps

    # model_flow.weight = Parameter_trans(model_flow.weight, model_torch.weight)
    # model_flow.bias = Parameter_trans(model_flow.bias, model_torch.bias)
    # model_flow.epsilon = model_torch.eps
    # model_flow.elementwise_affine = model_torch.elementwise_affine

    return model_flow


def Dropout_trans(model_flow, model_torch):
    print(" Dropout")
    assert isinstance(model_flow, flow.nn.Dropout)
    assert isinstance(model_torch, torch.nn.Dropout)

    # 似乎不需要？
    model_flow.p = model_torch.p

    return model_flow


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
    enter()
    assert isinstance(model_flow, roberta_flow.Roberta)
    assert isinstance(model_torch, transformers.RobertaModel)

    indent_msg("embeddings:")
    model_flow.embeddings = RobertaEmbeddings_trans(
        model_flow.embeddings, model_torch.embeddings)
    indent_msg("encoder:")
    model_flow.encoder = RobertaEncoder_trans(
        model_flow.encoder, model_torch.encoder)
    indent_msg("pooler:")
    model_flow.pooler = RobertaPooler_trans(
        model_flow.pooler, model_torch.pooler)

    quit()
    return model_flow
