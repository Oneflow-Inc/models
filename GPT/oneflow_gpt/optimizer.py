import oneflow as flow

from oneflow_gpt.model import Embedding, LayerNorm
from oneflow_gpt.model import ColumnParallelLinear, RowParallelLinear


def make_grad_scaler(args):
    if args.loss_scale is not None:
        grad_scaler = flow.amp.StaticGradScaler(args.loss_scale)
    elif args.initial_loss_scale is not None:
        grad_scaler = flow.amp.GradScaler(
            init_scale=args.initial_loss_scale,
            growth_factor=2.0,
            backoff_factor=0.5,
            growth_interval=args.loss_scale_window,
        )
    else:
        grad_scaler = None

    return grad_scaler


def make_lr_scheduler(args, optimizer):
    assert args.lr_decay_style in ("none", "cosine")

    if args.lr_decay_style == "none":
        return None

    if args.lr_decay_iters is None:
        return None

    lr_decay_alpha = args.min_lr / args.lr
    lr_scheduler = flow.optim.lr_scheduler.CosineDecayLR(
        optimizer, decay_steps=args.lr_decay_iters, alpha=lr_decay_alpha,
    )

    if args.lr_warmup_iters is not None and args.lr_warmup_iters > 0:
        lr_scheduler = flow.optim.lr_scheduler.WarmUpLR(
            lr_scheduler,
            warmup_factor=0,
            warmup_iters=args.lr_warmup_iters,
            warmup_method="linear",
        )

    return lr_scheduler


def _get_params_for_weight_decay_optimization(mode):
    weight_decay_params = {"params": []}
    no_weight_decay_params = {"params": [], "weight_decay": 0.0}

    for module in mode.modules():
        if isinstance(module, Embedding):
            for p in module.parameters(recurse=False):
                if p is None:
                    continue
                weight_decay_params["params"].append(p)
        elif isinstance(module, LayerNorm):
            for p in module.parameters(recurse=False):
                if p is None:
                    continue
                no_weight_decay_params["params"].append(p)
        elif isinstance(module, (ColumnParallelLinear, RowParallelLinear)):
            for n, p in module.named_parameters(recurse=False):
                if p is None:
                    continue
                if n == "weight":
                    weight_decay_params["params"].append(p)
                elif n == "bias":
                    no_weight_decay_params["params"].append(p)
                else:
                    raise NotImplementedError
        else:
            assert len(list(module.parameters(recurse=False))) == 0

    return [weight_decay_params, no_weight_decay_params]


def _set_clip_grad_for_param_groups(param_groups, clip_grad):
    if int(clip_grad) == 1:
        for group in param_groups:
            group["clip_grad_max_norm"] = 1.0
            group["clip_grad_norm_type"] = 2.0


def make_optimizer(args, model):
    param_groups = _get_params_for_weight_decay_optimization(model)
    _set_clip_grad_for_param_groups(param_groups, args.clip_grad)

    if args.optimizer == "adamw":
        optimizer = flow.optim.AdamW(
            param_groups,
            lr=args.lr,
            weight_decay=args.weight_decay,
            betas=(args.adam_beta1, args.adam_beta2),
            eps=args.adam_eps,
        )
    elif args.optimizer == "sgd":
        optimizer = flow.optim.SGD(param_groups, lr=args.lr)
    else:
        raise NotImplementedError("not supported yet")

    return optimizer
