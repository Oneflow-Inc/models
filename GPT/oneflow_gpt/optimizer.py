import oneflow as flow

from oneflow_gpt.model import LayerNorm


def make_lr_scheduler(args):
    # set up warmup strategy
    warmup = None
    if args.lr_warmup_iters is not None and args.lr_warmup_iters > 0:
        warmup = flow.optimizer.warmup.linear(args.lr_warmup_iters, 0)

    lr_decay_alpha = args.min_lr / args.lr
    # set up learning rate scheduler
    if args.lr_decay_style == "cosine" and args.lr_decay_iters is not None:
        lr_scheduler = flow.optimizer.CosineScheduler(
            base_lr=args.lr,
            steps=args.lr_decay_iters,
            alpha=lr_decay_alpha,
            warmup=warmup,
        )
    else:
        raise NotImplementedError("not supported yet")

    return lr_scheduler


def _get_params_for_weight_decay_optimization(mode):
    weight_decay_params = {"params": []}
    no_weight_decay_params = {"params": [], "weight_decay": 0.0}

    for module in mode.modules():
        if isinstance(module, LayerNorm):
            params = [p for p in module.parameters() if p is not None]
            no_weight_decay_params["params"].extend(params)
        else:
            for n, p in module.named_parameters():
                if p is None:
                    continue

                if n == "bias":
                    no_weight_decay_params["params"].append(p)
                else:
                    weight_decay_params["params"].append(p)

    return weight_decay_params, no_weight_decay_params


def _set_clip_grad_for_param_groups(param_groups, clip_grad):
    if int(clip_grad) == 1:
        for group in param_groups:
            group["clip_grad_max_norm"] = 1.0,
            group["clip_grad_norm_type"] = 2.0,
    else:
        raise NotImplementedError


def make_optimizer(args, model):
    param_groups = _get_params_for_weight_decay_optimization(model)
    _set_clip_grad_for_param_groups(param_groups, args.clip_grad)

    if args.optimizer == "adamw":
        adamw = flow.optim.AdamW(
            param_groups,
            lr=args.lr,
            weight_decay=args.weight_decay,
            betas=(args.adam_beta1, args.adam_beta2),
            eps=args.adam_eps,
        )
        return adamw
    else:
        raise NotImplementedError("not supported yet")
