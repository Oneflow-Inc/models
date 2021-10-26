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

from .distribute import get_model_parallel_group
from .utils import split_tensor_along_last_dim


def _reduce(input_):
    """All-reduce the the input tensor across model parallel group."""
    group = get_model_parallel_group()

    # Bypass the function if we are using only 1 GPU.
    if flow.distributed.get_world_size(group=group) == 1:
        return input_

    # All-reduce.
    flow.distributed.all_reduce(input_, group=group)

    return input_


def _split(input_):
    """Split the tensor along its last dimension and keep the
    corresponding slice."""
    group = get_model_parallel_group()

    # Bypass the function if we are using only 1 GPU.
    if flow.distributed.get_world_size(group=group) == 1:
        return input_

    # Split along last dimension.
    world_size = flow.distributed.get_world_size(group=group)
    input_list = split_tensor_along_last_dim(input_, world_size)

    # Note: flow.split does not create contiguous tensors by default.
    rank = flow.distributed.get_rank(group=group)
    output = input_list[rank].contiguous()

    return output


def _gather(input_):
    """Gather tensors and concatinate along the last dimension."""
    group = get_model_parallel_group()

    # Bypass the function if we are using only 1 GPU.
    if flow.distributed.get_world_size(group=group) == 1:
        return input_

    # Size and dimension.
    last_dim = input_.dim() - 1
    rank = flow.distributed.get_rank(group=group)
    world_size = flow.distributed.get_world_size(group=group)

    tensor_list = [flow.empty_like(input_) for _ in range(world_size)]
    tensor_list[rank] = input_
    flow.distributed.all_gather(tensor_list, input_, group=group)

    # Note: flow.cat already creates a contiguous tensor.
    output = flow.cat(tensor_list, dim=last_dim).contiguous()

    return output
