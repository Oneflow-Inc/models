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

from .initialize import get_model_parallel_group
from .initialize import get_model_parallel_rank
from .initialize import get_model_parallel_world_size
from .utils import VocabUtility


# class _VocabParallelCrossEntropy(flow.autograd.Function):

#     @staticmethod
#     def forward(ctx, vocab_parallel_logits, target):

#         logits = vocab_parallel_logits.clone()
       
#         logits_max = flow.max(logits, dim=-1)[0]
#         flow.distributed.all_reduce(logits_max,
#                                      op=flow.distributed.ReduceOp.MAX,
#                                      group=get_model_parallel_group())
       
#         logits.sub_(logits_max.unsqueeze(dim=-1))
      
#         exp_logits = logits.exp()
#         sum_exp_logits = exp_logits.sum(dim=-1)
#         flow.distributed.all_reduce(sum_exp_logits,
#                                      op=flow.distributed.ReduceOp.SUM,
#                                      group=get_model_parallel_group())

        
#         get_vocab_range = VocabUtility.vocab_range_from_per_partition_vocab_size
#         partition_vocab_size = vocab_parallel_logits.size()[-1]
#         rank = get_model_parallel_rank()
#         world_size = get_model_parallel_world_size()
#         vocab_start_index, vocab_end_index = get_vocab_range(
#             partition_vocab_size, rank, world_size)

#         target_mask = (target < vocab_start_index) | (target >= vocab_end_index)
#         masked_target = target.clone() - vocab_start_index
#         masked_target[target_mask] = 0

#         logits_2d = logits.view(-1, partition_vocab_size)
#         masked_target_1d = masked_target.view(-1)
#         arange_1d = flow._C.arange(start=0, end=logits_2d.size()[0],
#                                  device=logits_2d.device)
#         predicted_logits_1d = logits_2d[arange_1d, masked_target_1d]
#         predicted_logits = predicted_logits_1d.view_as(target)
#         predicted_logits[target_mask] = 0.0
       
#         flow.distributed.all_reduce(predicted_logits,
#                                      op=flow.distributed.ReduceOp.SUM,
#                                      group=get_model_parallel_group())

#         loss = flow.log(sum_exp_logits) - predicted_logits

#         exp_logits.div_(sum_exp_logits.unsqueeze(dim=-1))
#         ctx.save_for_backward(exp_logits, target_mask, masked_target_1d)

#         return loss

#     @staticmethod
#     def backward(ctx, grad_output):

#         softmax, target_mask, masked_target_1d = ctx.saved_tensors

#         grad_input = softmax
        
#         partition_vocab_size = softmax.size()[-1]
#         grad_2d = grad_input.view(-1, partition_vocab_size)

#         arange_1d = flow._C.arange(start=0, end=grad_2d.size()[0],
#                                  device=grad_2d.device)
#         grad_2d[arange_1d, masked_target_1d] -= (
#             1.0 - target_mask.view(-1).float())

#         grad_input.mul_(grad_output.unsqueeze(dim=-1))

#         return grad_input, None


def vocab_parallel_cross_entropy(vocab_parallel_logits, target):
    return _VocabParallelCrossEntropy.apply(vocab_parallel_logits, target)


def get_loss(vocab_parallel_logits,target):
    logits = vocab_parallel_logits.clone()
       
    logits_max = flow.max(logits, dim=-1)[0]
    
    # flow.distributed.all_reduce(logits_max,
    #                                 op=flow.distributed.ReduceOp.MAX,
    #                                 group=get_model_parallel_group())
    
    # logits.sub_(logits_max.unsqueeze(dim=-1))
    logits = logits - logits_max.unsqueeze(dim=-1)

    exp_logits = logits.exp()

    sum_exp_logits = exp_logits.sum(dim=-1)
    
    # flow.distributed.all_reduce(sum_exp_logits,
    #                                 op=flow.distributed.ReduceOp.SUM,
    #                                 group=get_model_parallel_group())

    get_vocab_range = VocabUtility.vocab_range_from_per_partition_vocab_size
    partition_vocab_size = vocab_parallel_logits.size()[-1]
    # rank = get_model_parallel_rank()
    # world_size = get_model_parallel_world_size()
    rank = 0
    world_size = 1
    vocab_start_index, vocab_end_index = get_vocab_range(
        partition_vocab_size, rank, world_size)
    
    target_mask = ((target < vocab_start_index) | (target >= vocab_end_index))
    masked_target = (target.clone() - vocab_start_index)
   
    masked_target[target_mask] = 0
    
    logits_2d = logits.view(-1, partition_vocab_size)

    masked_target_1d = masked_target.view(-1).to(flow.int)

    arange_1d = flow._C.arange(start=0, end=logits_2d.size()[0],
                                device=logits_2d.device).to(flow.int)
    predicted_logits_1d = logits_2d[arange_1d, masked_target_1d]
    predicted_logits = predicted_logits_1d.reshape(*target.size())
    predicted_logits[target_mask] = 0.0

    # flow.distributed.all_reduce(predicted_logits,
    #                                 op=flow.distributed.ReduceOp.SUM,
    #                                 group=get_model_parallel_group())

    loss = flow.log(sum_exp_logits) - predicted_logits

    # exp_logits.div_(sum_exp_logits.unsqueeze(dim=-1))
    # ctx.save_for_backward(exp_logits, target_mask, masked_target_1d)

    return loss
