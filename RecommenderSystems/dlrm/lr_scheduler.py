import os
import math
import oneflow as flow


def make_lr_scheduler(args, optimizer):
    warmup_lr = flow.optim.lr_scheduler.LinearLR(
    optimizer,
    start_factor=0,
    total_iters=args.warmup_batches,
    )
    poly_decay_lr = flow.optim.lr_scheduler.PolynomialLR(
    optimizer,
    steps=args.decay_batches,
    end_learning_rate=0,
    power=2.0,
    cycle=False,
    )
    sequential_lr = flow.optim.lr_scheduler.SequentialLR(
    optimizer=optimizer,
    schedulers=[warmup_lr, poly_decay_lr],
    milestones=[args.decay_start],
    interval_rescaling=True,
    )
    return sequential_lr
