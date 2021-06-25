#-*- coding:utf-8 -*-
""" 
 @author: scorpio.lu
 @datetime:2020-07-23 15:19
 @software: PyCharm
 @contact: luyi@zhejianglab.com

            ----------
             路有敬亭山
            ----------
 
"""
import oneflow.experimental as flow
from bisect import bisect_right
from typing import List
from oneflow.experimental.optim.lr_scheduler import LrScheduler

class WarmupMultiStepLR(LrScheduler):
    def __init__(
            self,
            optimizer: flow.optim.Optimizer,
            milestones: List[int],
            gamma: float = 0.1,
            warmup_factor: float = 0.001,
            warmup_iters: int = 1000,
            warmup_method: str = "linear",
            last_step: int = -1,
            **kwargs,
    ):
        if not list(milestones) == sorted(milestones):
            raise ValueError(
                "Milestones should be a list of" " increasing integers. Got {}", milestones
            )
        self.milestones = milestones
        self.gamma = gamma
        self.warmup_factor = warmup_factor
        self.warmup_iters = warmup_iters
        self.warmup_method = warmup_method
        super().__init__(optimizer, last_step)

    def get_lr(self) -> List[float]:
        warmup_factor = _get_warmup_factor_at_iter(
            self.warmup_method, self.last_step, self.warmup_iters, self.warmup_factor
        )
        # print('last_step:',self.last_step,'lr:',3.5e-04 * warmup_factor * self.gamma ** bisect_right(self.milestones, self.last_step))
        return [
            base_lr * warmup_factor * self.gamma ** bisect_right(self.milestones, self.last_step)
            for base_lr in self.base_lrs
        ]

    def _compute_values(self) -> List[float]:
        # The new interface
        return self.get_lr()



def _get_warmup_factor_at_iter(
        method: str, iter: int, warmup_iters: int, warmup_factor: float
) -> float:
    """
    Return the learning rate warmup factor at a specific iteration.
    See https://arxiv.org/abs/1706.02677 for more details.
    Args:
        method (str): warmup method; either "constant" or "linear".
        iter (int): iteration at which to calculate the warmup factor.
        warmup_iters (int): the number of warmup iterations.
        warmup_factor (float): the base warmup factor (the meaning changes according
            to the method used).
    Returns:
        float: the effective warmup factor at the given iteration.
    """
    if iter >= warmup_iters:
        return 1.0

    if method == "constant":
        return warmup_factor
    elif method == "linear":
        alpha = iter / warmup_iters
        return warmup_factor * (1 - alpha) + alpha
    else:
        raise ValueError("Unknown warmup method: {}".format(method))
