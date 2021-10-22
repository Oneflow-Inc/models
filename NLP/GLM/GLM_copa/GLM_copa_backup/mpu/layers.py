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


import math

import torch
import torch.nn.functional as F
import torch.nn.init as init
from torch.nn.parameter import Parameter

#from apex.normalization.fused_layer_norm import FusedLayerNorm as LayerNorm

from .initialize import get_model_parallel_rank
from .initialize import get_model_parallel_world_size
from .mappings import copy_to_model_parallel_region
from .mappings import gather_from_model_parallel_region
from .mappings import reduce_from_model_parallel_region
from .mappings import scatter_to_model_parallel_region
from .random import get_cuda_rng_tracker
from .utils import divide
from .utils import split_tensor_along_last_dim
from .utils import VocabUtility


def _initialize_affine_weight(weight, output_size, input_size,
                              per_partition_size, partition_dim, init_method,
                              stride=1, return_master_weight=False):

    # world_size = get_model_parallel_world_size()
    world_size = 1
    if world_size == 1:
        init_method(weight)
        if return_master_weight:
            return weight
        return None
    
    #下面不执行
    master_weight = torch.empty(output_size, input_size,
                                dtype=weight.dtype,
                                requires_grad=False)
    init_method(master_weight)

    per_partition_per_stride_size = divide(per_partition_size, stride)
    weight_list = torch.split(master_weight, per_partition_per_stride_size,
                              dim=partition_dim)
    rank = get_model_parallel_rank()
   
    my_weight_list = weight_list[rank::world_size]

    with torch.no_grad():
        torch.cat(my_weight_list, dim=partition_dim, out=weight)
    if return_master_weight:
        return master_weight
    return None


class VocabParallelEmbedding(torch.nn.Module):
   
    def __init__(self, num_embeddings, embedding_dim,
                 init_method=init.xavier_normal_):
        super(VocabParallelEmbedding, self).__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.padding_idx = None
        self.max_norm = None
        self.norm_type = 2.
        self.scale_grad_by_freq = False
        self.sparse = False
        self._weight = None
        self.vocab_start_index, self.vocab_end_index = \
            VocabUtility.vocab_range_from_global_vocab_size(
                self.num_embeddings, 0,1)
        self.num_embeddings_per_partition = self.vocab_end_index - \
                                            self.vocab_start_index

        self.weight = Parameter(torch.Tensor(self.num_embeddings_per_partition,
                                             self.embedding_dim))
        # self.weight.model_parallel = True
        _initialize_affine_weight(
            self.weight, self.num_embeddings, self.embedding_dim,
            self.num_embeddings_per_partition, 0, init_method)

    def forward(self, input_):
        ##不支持或操作符
        input_mask = (input_ < self.vocab_start_index) | \
                     (input_ >= self.vocab_end_index)
        # input_mask = torch._C.logical_or(input_ < self.vocab_start_index, input_ >= self.vocab_end_index)
        
        masked_input = (input_.clone() - self.vocab_start_index).to(torch.int)
        
        #不支持切片索引
        masked_input[input_mask] = 0

        output_parallel = F.embedding(masked_input, 
                                      self.weight,
                                      self.padding_idx, 
                                      self.max_norm,
                                      None,               #self.norm_type, 暂时变为none
                                      self.scale_grad_by_freq,
                                      self.sparse)
                                      
        #不支持切片索引
        output_parallel[input_mask, :] = 0.0
        # output_parallel[input_mask[...,None].expand(*output_parallel.shape).to(torch.int8)] = 0
   
        #output = reduce_from_model_parallel_region(output_parallel)
        output = output_parallel
        
        return output


class ParallelEmbedding(torch.nn.Module):
    def __init__(self, num_embeddings, embedding_dim,
                 init_method=init.xavier_normal_,
                 keep_master_weight_for_test=False):
        super(ParallelEmbedding, self).__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.padding_idx = None
        self.max_norm = None
        self.norm_type = 2.
        self.scale_grad_by_freq = False
        self.sparse = False
        self._weight = None
        world_size = get_model_parallel_world_size()
        self.embedding_dim_per_partition = divide(self.embedding_dim,
                                                  world_size)

        self.weight = Parameter(torch.Tensor(self.num_embeddings,
                                             self.embedding_dim_per_partition))
        self.weight.model_parallel = True
        _initialize_affine_weight(
            self.weight, self.num_embeddings, self.embedding_dim,
            self.embedding_dim_per_partition, 1, init_method,
            stride=1, return_master_weight=False)

    def forward(self, input_):
        input_parallel = copy_to_model_parallel_region(input_)
        output_parallel = F.embedding(input_parallel, self.weight,
                                      self.padding_idx, self.max_norm,
                                      self.norm_type, self.scale_grad_by_freq,
                                      self.sparse)
        output = gather_from_model_parallel_region(output_parallel)
        return output


class ColumnParallelLinear(torch.nn.Module):
    def __init__(self, input_size, output_size, bias=True, gather_output=True,
                 init_method=init.xavier_normal_, stride=1,
                 keep_master_weight_for_test=False):
        super(ColumnParallelLinear, self).__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.gather_output = gather_output
        
        world_size = 1
        # world_size = get_model_parallel_world_size()

        self.output_size_per_partition = divide(output_size, world_size)

        self.weight = Parameter(torch.Tensor(self.output_size_per_partition,
                                             self.input_size))
        # self.weight.model_parallel = True
        if bias:
            self.bias = Parameter(torch.Tensor(self.output_size_per_partition))

            # self.bias.model_parallel = True
            # with torch.no_grad():
            #     self.bias.zero_()
        else:
            self.register_parameter('bias', None)

        self.master_weight = _initialize_affine_weight(
            self.weight, self.output_size, self.input_size,
            self.output_size_per_partition, 0, init_method,
            stride=stride, return_master_weight=keep_master_weight_for_test)

    def forward(self, input_):

        #input_parallel = copy_to_model_parallel_region(input_)
        input_parallel = input_

        output_parallel = F.linear(input_parallel, self.weight, self.bias)
        # if self.gather_output:
        #     output = gather_from_model_parallel_region(output_parallel)
        # else:
        #     output = output_parallel
        return output_parallel


class RowParallelLinear(torch.nn.Module):
    def __init__(self, input_size, output_size, bias=True,
                 input_is_parallel=False,
                 init_method=init.xavier_normal_, stride=1,
                 keep_master_weight_for_test=False):
        super(RowParallelLinear, self).__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.input_is_parallel = input_is_parallel

        world_size = 1
        # world_size = get_model_parallel_world_size()

        self.input_size_per_partition = divide(input_size, world_size)

    
        self.weight = Parameter(torch.Tensor(self.output_size,
                                             self.input_size_per_partition))
        # self.weight.model_parallel = True

        if bias:
            self.bias = Parameter(torch.Tensor(self.output_size))
            # with torch.no_grad():
            #     self.bias.zero_()
        else:
            self.register_parameter('bias', None)
        self.master_weight = _initialize_affine_weight(
            self.weight, self.output_size, self.input_size,
            self.input_size_per_partition, 1, init_method,
            stride=stride, return_master_weight=keep_master_weight_for_test)

    def forward(self, input_):
        
        # if self.input_is_parallel:
        #     input_parallel = input_
        # else:
        #     input_parallel = scatter_to_model_parallel_region(input_)
      
        output_parallel = F.linear(input_, self.weight)
      
        # output_ = reduce_from_model_parallel_region(output_parallel)
        if self.bias is not None:
            output = output_parallel + self.bias
        else:
            output = output_parallel
        return output

