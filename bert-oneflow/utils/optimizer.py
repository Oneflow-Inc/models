import oneflow.nn as nn
import copy
import oneflow as flow
from typing import List
from .lamb_optimizer import LAMB


def build_lamb_optimizer(
    model: nn.Module, lr: float, weight_decay: float, weight_decay_excludes: List[str],
) -> flow.optim.Optimizer:
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

    return LAMB(params, betas=(0.9, 0.999), eps=1e-6)


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
    model: nn.Module,
    lr: float,
    momentum: float,
    weight_decay: float,
    weight_decay_excludes: List[str] = [""],
    clip_grad_max_norm: float = None,
    clip_grad_norm_type: float = None,
) -> flow.optim.Optimizer:
    defaults = {"lr": lr, "momentum": momentum}
    params = []
    exclude_params = []
    use_clip = clip_grad_max_norm is not None and clip_grad_norm_type is not None
    hyperparameters = copy.copy(defaults)
    for module_name, param in model.named_parameters():
        add_to_exclude = False
        for exclude_name in weight_decay_excludes:
            if module_name.find(exclude_name) != -1:
                exclude_params.append(param)
                add_to_exclude = True
                break
        if not add_to_exclude:
            params.append(param)

    if use_clip:
        all_params = [
            {
                "params": params,
                "clip_grad_max_norm": clip_grad_max_norm,
                "clip_grad_norm_type": clip_grad_norm_type,
                "weight_decay": weight_decay,
                **hyperparameters,
            },
            {
                "params": exclude_params,
                "clip_grad_max_norm": clip_grad_max_norm,
                "clip_grad_norm_type": clip_grad_norm_type,
                "weight_decay": 0.0,
                **hyperparameters,
            },
        ]
    else:
        all_params = [
            {"params": params, "weight_decay": weight_decay, **hyperparameters,},
            {"params": exclude_params, "weight_decay": 0.0, **hyperparameters,},
        ]

    return flow.optim.SGD(all_params)
