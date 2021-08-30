import sys

import oneflow as flow
import torch

sys.path.append("../")
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