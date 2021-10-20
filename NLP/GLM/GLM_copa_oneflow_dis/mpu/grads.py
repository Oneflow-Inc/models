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


# Parts of the code here are adapted from PyTorch
# repo: https://github.com/pytorch/pytorch


import oneflow as flow
# from oneflow._six import inf

# from .initialize import get_model_parallel_group
# from .initialize import get_model_parallel_rank


def clip_grad_norm(parameters, max_norm, norm_type=2):
    return 
    if isinstance(parameters, flow.Tensor):
        parameters = [parameters]
    parameters = list(filter(lambda p: p.grad is not None, parameters))
    max_norm = float(max_norm)
    norm_type = float(norm_type)
    # if norm_type == inf:
    #     total_norm = max(p.grad.data.abs().max() for p in parameters)
    #     total_norm_cuda = flow.cuda.FloatTensor([float(total_norm)])
       
    #     # flow.distributed.all_reduce(total_norm_cuda,
    #     #                              op=flow.distributed.ReduceOp.MAX,
    #     #                              group=get_model_parallel_group())
    #     total_norm = total_norm_cuda[0].item()
    # else:
    total_norm = 0
    for p in parameters:
        param_norm = p.grad.data.norm(norm_type)
        total_norm += param_norm.item() ** norm_type
    
    total_norm_cuda = torch.cuda.FloatTensor([float(total_norm)])
    # torch.distributed.all_reduce(total_norm_cuda,
    #                              op=torch.distributed.ReduceOp.SUM,
    #                              group=get_model_parallel_group())
    total_norm = total_norm_cuda[0].item() ** (1. / norm_type)
    
    clip_coef = max_norm / (total_norm + 1e-6)
    if clip_coef < 1:
        for p in parameters:
            p.grad.data.mul_(clip_coef)
    return total_norm
