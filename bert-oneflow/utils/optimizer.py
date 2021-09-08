import oneflow.nn as nn
import copy
import oneflow as flow
from typing import List


def build_adamW_optimizer(
    model: nn.Module, lr: float, weight_decay: float, weight_decay_excludes: List[str],
):
    defaults = {"lr": lr, "weight_decay": weight_decay}
    params = []
    for module_param_name, value in model.named_parameters():
        if not value.requires_grad:
            continue

        hyperparameters = copy.copy(defaults)
        for exclude_name in weight_decay_excludes:
            if module_param_name.find(exclude_name) != -1:
                hyperparameters["weight_decay"] = 0
                break

        params.append({"params": [value], **hyperparameters})

    return flow.optim.AdamW(params)


def build_sgd_optimizer(
    model: nn.Module, lr: float, momentum: float,
):
    defaults = {"lr": lr, "momentum": momentum}
    params = []
    for module_param_name, value in model.named_parameters():
        if not value.requires_grad:
            continue
        hyperparameters = copy.copy(defaults)
        params.append({"params": [value], **hyperparameters})

    return flow.optim.SGD(params)
