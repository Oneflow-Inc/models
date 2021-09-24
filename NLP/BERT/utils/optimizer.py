import oneflow.nn as nn
import copy
import oneflow as flow
from typing import List
from .lamb_optimizer import LAMB


def build_optimizer(
    optim_name: str,
    model: nn.Module,
    lr: float,
    weight_decay: float,
    weight_decay_excludes: List[str],
    clip_grad_max_norm: float = None,
    clip_grad_norm_type: float = None,
):
    assert optim_name in [
        "adamw",
        "lamb",
    ], f"only support adamw and lamb now, but got {optim_name}"

    params = []
    exclude_params = []
    use_clip = clip_grad_max_norm is not None and clip_grad_norm_type is not None
    if use_clip:
        defaults = {
            "lr": lr,
            "clip_grad_max_norm": clip_grad_max_norm,
            "clip_grad_norm_type": clip_grad_norm_type,
        }
    else:
        defaults = {"lr": lr}
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

    all_params = [
        {"params": params, "weight_decay": weight_decay, **hyperparameters,},
        {"params": exclude_params, "weight_decay": 0.0, **hyperparameters,},
    ]

    if optim_name == "adamw":
        return flow.optim.AdamW(all_params)
    elif optim_name == "lamb":
        return LAMB(all_params)


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
