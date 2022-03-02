# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# LARS optimizer, implementation from MoCo v3:
# https://github.com/facebookresearch/moco-v3
# --------------------------------------------------------

# import torch
import oneflow as flow


class LARS(flow.optim.Optimizer):
    """
    LARS optimizer, no rate scaling or weight decay for parameters <= 1D.
    """
    def __init__(self, params, lr=0, weight_decay=0, momentum=0.9, trust_coefficient=0.001):
        defaults = dict(lr=lr, weight_decay=weight_decay, momentum=momentum, trust_coefficient=trust_coefficient)
        super().__init__(params, defaults)
        for param_group in self.param_groups:
            for param in param_group.parameters:
                assert param.is_leaf, "parameters must be leaf tensor"
                self._state[param] = dict()

    
    def step(self):
        with flow.no_grad():
            for g in self.param_groups:
                for p in g.parameters:
                    dp = p.grad

                    if dp is None:
                        continue

                    if p.ndim > 1: # if not normalization gamma/beta or bias
                        dp = dp.add(p*g['weight_decay'])
                        param_norm = flow.linalg.norm(p)
                        update_norm = flow.linalg.norm(dp)
                        one = flow.ones_like(param_norm)
                        q = flow.where(param_norm > 0.,
                                        flow.where(update_norm > 0,
                                        (g['trust_coefficient'] * param_norm / update_norm), one),
                                        one)
                        dp = dp.mul(q)

                    param_state = self._state[p]
                    if 'mu' not in param_state:
                        param_state['mu'] = flow.zeros_like(p)
                    mu = param_state['mu']
                    mu.mul_(g['momentum']).add_(dp)
                    p.add_(-mu*g['lr'])