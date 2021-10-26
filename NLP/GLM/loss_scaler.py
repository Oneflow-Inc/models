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

import oneflow as flow

def to_python_float(t):
    if hasattr(t, 'item'):
        return t.item()
    else:
        return t[0]

class LossScaler:
    def __init__(self, scale=1):
        self.cur_scale = scale

    def has_overflow(self, params):
        return False

    def _has_inf_or_nan(x):
        return False

    def update_scale(self, overflow):
        pass

    @property
    def loss_scale(self):
        return self.cur_scale

    def scale_gradient(self, module, grad_in, grad_out):
        return tuple(self.loss_scale * g for g in grad_in)

    def backward(self, loss, retain_graph=False):
        scaled_loss = loss*self.loss_scale
        scaled_loss.backward(retain_graph=retain_graph)

class DynamicLossScaler:
    def __init__(self,
                 init_scale=2**32,
                 scale_factor=2.,
                 scale_window=1000,
                 min_scale=1,
                 delayed_shift=1,
                 consecutive_hysteresis=False):
        self.cur_scale = init_scale
        self.cur_iter = 0
        self.last_overflow_iter = -1
        self.scale_factor = scale_factor
        self.scale_window = scale_window
        self.min_scale = min_scale
        self.delayed_shift = delayed_shift
        self.cur_hysteresis = delayed_shift
        self.consecutive_hysteresis = consecutive_hysteresis


    def _has_inf_or_nan(x):
        try:
            cpu_sum = float(x.float().sum().item())
        except RuntimeError as instance:
            if "value cannot be converted" not in instance.args[0]:
                raise
            return True
        else:
            if cpu_sum == float('inf') or cpu_sum == -float('inf') or cpu_sum != cpu_sum:
                return True
            return False

    def update_scale(self, overflow):

        if not hasattr(self, 'min_scale'):
            self.min_scale = 1
        if not hasattr(self, 'delayed_shift'):
            self.delayed_shift = 1
        if not hasattr(self, 'cur_hysteresis'):
            self.cur_hysteresis = 1
        if not hasattr(self, 'consecutive_hysteresis'):
            self.consecutive_hysteresis = True
        if overflow:
            if self.delayed_shift == 1 or self.cur_hysteresis == 1:
                self.cur_scale = max(self.cur_scale/self.scale_factor, self.min_scale)
            else:
                self.cur_hysteresis -= 1
            self.last_overflow_iter = self.cur_iter
        else:
            if self.consecutive_hysteresis:
                self.cur_hysteresis = self.delayed_shift
            if (self.cur_iter - self.last_overflow_iter) % self.scale_window == 0:
                if not self.consecutive_hysteresis:
                    self.cur_hysteresis = self.delayed_shift
                self.cur_scale *= self.scale_factor
        self.cur_iter += 1

    @property
    def loss_scale(self):
        return self.cur_scale

    def scale_gradient(self, module, grad_in, grad_out):
        return tuple(self.loss_scale * g for g in grad_in)

    def backward(self, loss, retain_graph=False):
        scaled_loss = loss*self.loss_scale
        scaled_loss.backward(retain_graph=retain_graph)
