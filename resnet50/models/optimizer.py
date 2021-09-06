import oneflow as flow


def make_optimizer(args, model):
    param_group = {"params": [p for p in model.parameters() if p is not None]}

    if args.grad_clipping > 0.0:
        assert args.grad_clipping == 1.0, "ONLY support grad_clipping == 1.0"
        param_group["clip_grad_max_norm"] = (1.0,)
        param_group["clip_grad_norm_type"] = (2.0,)

    optimizer = flow.optim.SGD(
        [param_group],
        lr=args.learning_rate,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )
    return optimizer


def make_grad_scaler():
    return flow.amp.GradScaler(
        init_scale=2 ** 30, growth_factor=2.0, backoff_factor=0.5, growth_interval=2000,
    )


def make_static_grad_scaler():
    return flow.amp.StaticGradScaler(flow.env.get_world_size())


def make_lr_scheduler(args, optimizer):
    assert args.lr_decay_type in ("none", "cosine")

    if args.lr_decay_type == "none":
        return None

    warmup_batches = args.batches_per_epoch * args.warmup_epochs
    total_batches = args.batches_per_epoch * args.num_epochs
    # TODO(zwx): These's no need that decay_batches minus warmup_batches
    # decay_batches = total_batches - warmup_batches
    decay_batches = total_batches

    lr_scheduler = flow.optim.lr_scheduler.CosineDecayLR(
        optimizer, decay_steps=decay_batches
    )

    if args.warmup_epochs > 0:
        lr_scheduler = flow.optim.lr_scheduler.WarmUpLR(
            lr_scheduler,
            warmup_factor=0,
            warmup_iters=warmup_batches,
            warmup_method="linear",
        )

    return lr_scheduler


def make_cross_entropy(args):
    if args.label_smoothing > 0:
        cross_entropy = LabelSmoothLoss(
            num_classes=args.num_classes, smooth_rate=args.label_smoothing
        )
    else:
        cross_entropy = flow.nn.CrossEntropyLoss(reduction="mean")

    return cross_entropy


class LabelSmoothLoss(flow.nn.Module):
    def __init__(self, num_classes=-1, smooth_rate=0.0):
        super().__init__()
        self.num_classes = num_classes
        self.smooth_rate = smooth_rate
        # TODO(zwx): check this hyper param correction
        self.on_value = 1 - self.smooth_rate + self.smooth_rate / self.num_classes
        self.off_value = self.smooth_rate / self.num_classes

    def forward(self, input, label):
        onehot_label = flow._C.one_hot(
            label, self.num_classes, self.on_value, self.off_value
        )
        # NOTE(zwx): manual way has bug
        # log_prob = input.softmax(dim=-1).log()
        # onehot_label = flow.F.cast(onehot_label, log_prob.dtype)
        # loss = flow.mul(log_prob * -1, onehot_label).sum(dim=-1).mean()
        loss = flow._C.softmax_cross_entropy(input, onehot_label.to(dtype=input.dtype))
        return loss.mean()
