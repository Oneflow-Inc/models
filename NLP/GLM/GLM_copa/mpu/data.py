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

import torch

from .initialize import get_model_parallel_group
from .initialize import get_model_parallel_rank
from .initialize import get_model_parallel_src_rank


_MAX_DATA_DIM = 5


def _check_data_types(keys, data, target_dtype):
    for key in keys:
        assert data[key].dtype == target_dtype, '{} has data type {} which '\
            'is different than {}'.format(key, data[key].dtype, target_dtype)


def _build_key_size_numel_dictionaries(keys, data):
    max_dim = _MAX_DATA_DIM
    sizes = [0 for _ in range(max_dim) for _ in keys]
    
    #True
    if get_model_parallel_rank() == 0:
        offset = 0
        for key in keys:
            assert data[key].dim() < max_dim, 'you should increase MAX_DATA_DIM'
            size = data[key].size()
            for i, s in enumerate(size):
                sizes[i + offset] = s
            offset += max_dim
    

    sizes_cuda = torch.cuda.LongTensor(sizes)
    torch.distributed.broadcast(sizes_cuda, get_model_parallel_src_rank(),
                                group=get_model_parallel_group())

    sizes_cpu = sizes_cuda.cpu()

    key_size = {}
    key_numel = {}
    total_numel = 0
    offset = 0
    for key in keys:
        i = 0
        size = []
        numel = 1
        while sizes_cpu[offset + i] > 0:
            this_size = sizes_cpu[offset + i]
            size.append(this_size)
            numel *= this_size
            i += 1
        key_size[key] = size
        key_numel[key] = numel
        total_numel += numel
        offset += max_dim

    return key_size, key_numel, total_numel


def broadcast_data(keys, data, datatype):
    key_size, key_numel, total_numel = _build_key_size_numel_dictionaries(keys,
                                                            data)
    #True
    if get_model_parallel_rank() == 0:
        _check_data_types(keys, data, datatype)
        flatten_data = torch.cat(
            [data[key].contiguous().view(-1) for key in keys], dim=0).cuda()
    else:
        flatten_data = torch.empty(total_numel,
                                   device=torch.cuda.current_device(),
                                   dtype=datatype)

    torch.distributed.broadcast(flatten_data, get_model_parallel_src_rank(),
                                group=get_model_parallel_group())

    output = {}
    offset = 0
    for key in keys:
        size = key_size[key]
        numel = key_numel[key]
        output[key] = flatten_data.narrow(0, offset, numel).view(size)
        offset += numel

    return output
