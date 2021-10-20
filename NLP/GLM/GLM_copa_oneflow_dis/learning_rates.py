# coding=utf-8
# Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""PyTorch DataLoader for TFRecords"""

import oneflow as flow
from oneflow.optim.lr_scheduler import _LRScheduler
import math


class AnnealingLR(_LRScheduler):
    DECAY_STYLES = ['linear', 'cosine', 'exponential', 'constant', 'None']

    def __init__(self, optimizer, start_lr, warmup_iter, num_iters, decay_style=None, last_iter=-1, decay_ratio=0.5):
        assert warmup_iter <= num_iters
        self._optimizer = optimizer
        self.start_lr = start_lr
        self.warmup_iter = warmup_iter
        self.num_iters = last_iter + 1
        self.end_iter = num_iters
        self.decay_style = decay_style.lower() if isinstance(decay_style, str) else None

        self.decay_ratio = 1 / decay_ratio
        self.step(self.num_iters)
        super().__init__(optimizer, 100000, False)
        # if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
        #     print(f'learning rate decaying style {self.decay_style}, ratio {self.decay_ratio}')

    def get_lr(self):
        if self.warmup_iter > 0 and self.num_iters <= self.warmup_iter:
            return float(self.start_lr) * self.num_iters / self.warmup_iter
        else:
            if self.decay_style == self.DECAY_STYLES[0]:
                decay_step_ratio = (self.num_iters - self.warmup_iter) / self.end_iter
                return self.start_lr - self.start_lr * (1 - 1 / self.decay_ratio) * decay_step_ratio
            elif self.decay_style == self.DECAY_STYLES[1]:
                decay_step_ratio = min(1.0, (self.num_iters - self.warmup_iter) / self.end_iter)
                return self.start_lr / self.decay_ratio * (
                        (math.cos(math.pi * decay_step_ratio) + 1) * (self.decay_ratio - 1) / 2 + 1)
            elif self.decay_style == self.DECAY_STYLES[2]:
                return self.start_lr
            else:
                return self.start_lr

    def step(self, step_num=None):
        if step_num is None:
            step_num = self.num_iters + 1
        self.num_iters = step_num
        new_lr = self.get_lr()
    
        for group in self._optimizer.param_groups:
            group['lr'] = new_lr

    def state_dict(self):
        sd = {
            # 'start_lr': self.start_lr,
            'warmup_iter': self.warmup_iter,
            'num_iters': self.num_iters,
            'decay_style': self.decay_style,
            'end_iter': self.end_iter,
            'decay_ratio': self.decay_ratio
        }
        return sd

    def load_state_dict(self, sd):
        self.warmup_iter = sd['warmup_iter']
        self.num_iters = sd['num_iters']
        self.step(self.num_iters)

    def switch_linear(self, args):
        current_lr = self.get_lr()
        self.start_lr = current_lr
        self.end_iter = args.train_iters - self.num_iters
        self.num_iters = 0
        self.decay_style = "linear"
    
    def _generate_conf_for_graph(self, opt_confs):
        # CosineDecayLR is the same as CosineDecayConf in nn.Graph
        for opt_conf in opt_confs:
            learning_rate_decay_conf = opt_conf.mutable_learning_rate_decay()
            learning_rate_decay_conf.mutable_polynomial_conf().set_decay_batches(
                self.max_decay_steps
            )
            learning_rate_decay_conf.mutable_polynomial_conf().set_end_learning_rate(
                self.end_learning_rate
            )
            learning_rate_decay_conf.mutable_polynomial_conf().set_power(self.power)
            learning_rate_decay_conf.mutable_polynomial_conf().set_cycle(self.cycle)