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

"""Transformer."""

import math

import oneflow as flow
import oneflow.nn.init as init
from  oneflow.nn import LayerNorm

from .distribute import get_model_parallel_world_size
from .layers import ColumnParallelLinear
from .layers import RowParallelLinear

import deepspeed


from .utils import divide
from .utils import split_tensor_along_last_dim


class PositionalEmbedding(flow.nn.Module):
    def __init__(self, hidden_size):
        super(PositionalEmbedding, self).__init__()

        self.hidden_size = hidden_size

        inv_freq = 1 / (10000 ** (flow._C.arange(0.0, hidden_size, 2.0) / hidden_size))
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, pos_seq, bsz=None):
        sinusoid_inp = flow.ger(pos_seq, self.inv_freq)
        pos_emb = flow.cat([sinusoid_inp.sin(), sinusoid_inp.cos()], dim=-1)

        if bsz is not None:
            return pos_emb[None, :, :].expand(bsz, -1, -1)
        else:
            return pos_emb[None, :, :]


class ParallelCrossAttention(flow.nn.Module):

    def __init__(self, hidden_size, num_attention_heads, attention_dropout_prob, output_dropout_prob, init_method,
                 output_layer_init_method=None):
        super(ParallelCrossAttention, self).__init__()
      
        if output_layer_init_method is None:
            output_layer_init_method = init_method
       
        world_size = get_model_parallel_world_size()
        self.hidden_size_per_partition = divide(hidden_size, world_size)
        self.hidden_size_per_attention_head = divide(hidden_size,
                                                     num_attention_heads)
        self.num_attention_heads_per_partition = divide(num_attention_heads,
                                                        world_size)
      
        self.query = ColumnParallelLinear(hidden_size, hidden_size,
                                          gather_output=False,
                                          init_method=init_method, 
                                          if_use_gelu=False, 
                                          if_use_dropout=False
                                          )
        self.key_value = ColumnParallelLinear(hidden_size, 2 * hidden_size,
                                              stride=2,
                                              gather_output=False,
                                              init_method=init_method, 
                                              if_use_gelu=False, 
                                              if_use_dropout=False)
      
        self.attention_dropout = flow.nn.Dropout(attention_dropout_prob)


        # self.dense = RowParallelLinear(hidden_size,
        #                                hidden_size,
        #                                input_is_parallel=True,
        #                                init_method=output_layer_init_method)
        # self.output_dropout = flow.nn.Dropout(output_dropout_prob)


        self.dense = RowParallelLinear(hidden_size,
                                       hidden_size,
                                       input_is_parallel=True,
                                       init_method=output_layer_init_method, 
                                       if_use_dropout=True, 
                                       dropout_rate=output_dropout_prob)

    def _transpose_for_scores(self, tensor):
        new_tensor_shape = tensor.size()[:-1] + \
                           (self.num_attention_heads_per_partition,
                            self.hidden_size_per_attention_head)
        tensor = tensor.view(*new_tensor_shape)
        return tensor.permute(0, 2, 1, 3)

    def forward(self, hidden_states, encoder_states, cross_mask):
        mixed_query_layer = self.query(hidden_states)
        mixed_x_layer = self.key_value(encoder_states)
        (mixed_key_layer, mixed_value_layer) = split_tensor_along_last_dim(mixed_x_layer, 2)

        query_layer = self._transpose_for_scores(mixed_query_layer)
        key_layer = self._transpose_for_scores(mixed_key_layer)
        value_layer = self._transpose_for_scores(mixed_value_layer)
    
        attention_scores = flow.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(
            self.hidden_size_per_attention_head)
        if cross_mask is not None:
            attention_scores = flow.mul(attention_scores, cross_mask) - \
                               10000.0 * (1.0 - cross_mask)

        attention_probs = flow.nn.Softmax(dim=-1)(attention_scores)
        attention_probs = self.attention_dropout(attention_probs)

        context_layer = flow.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + \
                                  (self.hidden_size_per_partition,)
        context_layer = context_layer.view(*new_context_layer_shape)

        # Previous
        # output = self.dense(context_layer)
        # output = self.output_dropout(output)

        # Fused Bias add and Dropout
        output = self.dense(context_layer)

        return output


class ParallelSelfAttention(flow.nn.Module):

    def __init__(self, hidden_size, num_attention_heads,
                 attention_dropout_prob, output_dropout_prob,
                 init_method, output_layer_init_method=None, relative_encoding=False,
                 performer=False, attention_scale=1.0):
        super(ParallelSelfAttention, self).__init__()
        self.performer = performer

        if output_layer_init_method is None:
            output_layer_init_method = init_method
        
        world_size = 1
        # world_size = get_model_parallel_world_size()

        self.hidden_size_per_partition = divide(hidden_size, world_size)
        self.hidden_size_per_attention_head = divide(hidden_size,
                                                     num_attention_heads)
        self.num_attention_heads_per_partition = divide(num_attention_heads,
                                                        world_size)
        self.relative_encoding = relative_encoding
        self.attention_scale = attention_scale
        
        self.query_key_value = ColumnParallelLinear(hidden_size, 3 * hidden_size,
                                                    stride=3,
                                                    gather_output=False,
                                                    init_method=init_method)
        if relative_encoding:
            self.relative = ColumnParallelLinear(hidden_size, hidden_size, gather_output=False,
                                                 init_method=init_method)
       
        self.attention_dropout = flow.nn.Dropout(attention_dropout_prob)

        # self.dense = RowParallelLinear(hidden_size,
                                    #    hidden_size,
                                    #    input_is_parallel=True,
                                    #    init_method=output_layer_init_method)
        # self.output_dropout = flow.nn.Dropout(output_dropout_prob)

        self.dense = RowParallelLinear(hidden_size,
                                       hidden_size,
                                       input_is_parallel=True,
                                       init_method=output_layer_init_method, 
                                       if_use_dropout=True, 
                                       dropout_rate=output_dropout_prob)

    def _transpose_for_scores(self, tensor):
        
        #不支持
        # new_tensor_shape = tensor.size()[:-1] + \
        #                    (self.num_attention_heads_per_partition,
        #                     self.hidden_size_per_attention_head)
        # new_tensor_shape = [*tensor.size()[:-1],self.num_attention_heads_per_partition,self.hidden_size_per_attention_head]
        size = tensor.size()
        new_tensor_shape = [size[0], size[1], self.num_attention_heads_per_partition, self.hidden_size_per_attention_head]
        tensor = tensor.reshape(*new_tensor_shape)
        return tensor.permute(0, 2, 1, 3)

    @staticmethod
    def _rel_shift(x, zero_triu=False):
        zero_pad = flow.zeros((*x.size()[:-2], x.size(-2), 1),
                               device=x.device, dtype=x.dtype)
        x_padded = flow.cat([zero_pad, x], dim=-1)

        x_padded = x_padded.view(*x.size()[:-2], x.size(-1) + 1, x.size(-2))

        x = x_padded[:, :, 1:].view_as(x)

        if zero_triu:
            ones = flow.ones((x.size(0), x.size(1)))
            x = x * flow.tril(ones, x.size(1) - x.size(0))[:, :, None, None]

        return x

    def forward(self, hidden_states, ltor_mask, position_embeddings=None, r_w_bias=None, r_r_bias=None, mem=None):
        query_length = hidden_states.size(1)
        
        #True
        if mem is None:
            mixed_x_layer = self.query_key_value(hidden_states)
            (mixed_query_layer,
             mixed_key_layer,
             mixed_value_layer) = split_tensor_along_last_dim(mixed_x_layer, 3)
        else:
            cat = flow.cat((mem, hidden_states), 1)
            mixed_x_layer = self.query_key_value(cat)
            (mixed_query_layer,
             mixed_key_layer,
             mixed_value_layer) = split_tensor_along_last_dim(mixed_x_layer, 3)
            mixed_query_layer = mixed_query_layer[:, -query_length:]
        

        query_layer = self._transpose_for_scores(mixed_query_layer)

        key_layer = self._transpose_for_scores(mixed_key_layer)

        value_layer = self._transpose_for_scores(mixed_value_layer)
        #False
        if self.relative_encoding:
            relative_layer = self.relative(position_embeddings)
            relative_layer = self._transpose_for_scores(relative_layer) 
            rw_head_q = query_layer + r_w_bias.unsqueeze(1)
            ac_score = flow.matmul(rw_head_q, key_layer.transpose(-1, -2))
            rr_head_q = query_layer + r_r_bias.unsqueeze(1)
            bd_score = flow.matmul(rr_head_q, relative_layer.transpose(-1, -2))
            bd_score = self._rel_shift(bd_score)  

            attention_scores = ac_score + bd_score
            attention_scores = attention_scores / math.sqrt(self.hidden_size_per_attention_head)
        else:
            if self.attention_scale > 1.0:
                attention_scores = flow.matmul(query_layer / math.sqrt(self.attention_scale),
                                            key_layer.transpose(-1, -2) / math.sqrt(
                                                self.hidden_size_per_attention_head * self.attention_scale))
            else:
                attention_scores = flow.matmul(query_layer, key_layer.transpose(-1, -2) / math.sqrt(
                    self.hidden_size_per_attention_head))

        
        attention_scores = flow.mul(attention_scores, ltor_mask)
        if self.attention_scale > 1.0:
            max_attention_scores = attention_scores.max(dim=-1, keepdim=True)[0]
            attention_scores -= max_attention_scores
            attention_scores *= self.attention_scale

        attention_scores = attention_scores + (-65504.0) * (1.0 - ltor_mask)
    
        attention_probs = flow.nn.Softmax(dim=-1)(attention_scores)

        attention_probs = self.attention_dropout(attention_probs)
    
        context_layer = flow.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        
        # new_context_layer_shape = context_layer.size()[:-2] + \
        #                           (self.hidden_size_per_partition,)
        new_context_layer_shape = [*context_layer.size()[:-2],self.hidden_size_per_partition]
        context_layer = context_layer.reshape(*new_context_layer_shape)
        
        # Previous
        # output = self.dense(context_layer)
        # output = self.output_dropout(output)

        # Fused bias add and Dropout
        output = self.dense(context_layer)

        return output


# @torch.jit.script
def gelu_impl(x):
    return 0.5 * x * (1.0 + flow.tanh(0.7978845608028654 * x *
                                       (1.0 + 0.044715 * x * x)))


def gelu(x):
    return gelu_impl(x)
    # return flow._C.gelu(x)


class ParallelMLP(flow.nn.Module):

    def __init__(self, hidden_size, output_dropout_prob, init_method,
                 output_layer_init_method=None):
        super(ParallelMLP, self).__init__()
     
        if output_layer_init_method is None:
            output_layer_init_method = init_method
       
        # self.dense_h_to_4h = ColumnParallelLinear(hidden_size, 4 * hidden_size,
        #                                           gather_output=False,
        #                                           init_method=init_method)

        self.dense_h_to_4h = ColumnParallelLinear(hidden_size, 4 * hidden_size,
                                                  gather_output=False,
                                                  init_method=init_method, 
                                                  if_use_gelu=True)

        # self.dense_4h_to_h = RowParallelLinear(
            # 4 * hidden_size,
            # hidden_size,
            # input_is_parallel=True,
            # init_method=output_layer_init_method)
        # self.dropout = flow.nn.Dropout(output_dropout_prob)

        self.dense_4h_to_h = RowParallelLinear(
            4 * hidden_size,
            hidden_size,
            input_is_parallel=True,
            init_method=output_layer_init_method, 
            if_use_dropout=True, 
            dropout_rate=output_dropout_prob)

    def forward(self, hidden_states):
        # previous
        # intermediate_parallel = self.dense_h_to_4h(hidden_states)
        # intermediate_parallel = gelu(intermediate_parallel)
        
        # Fused bias add and Gelu
        intermediate_parallel = self.dense_h_to_4h(hidden_states)


        # previous
        # output = self.dense_4h_to_h(intermediate_parallel)
        # output = self.dropout(output)
        
        # Fused bias add and Dropout
        output = self.dense_4h_to_h(intermediate_parallel)

        return output


class ParallelDecoderLayer(flow.nn.Module):
    
    def __init__(self,
                 hidden_size,
                 num_attention_heads,
                 attention_dropout_prob,
                 output_dropout_prob,
                 layernorm_epsilon,
                 init_method,
                 output_layer_init_method=None):
        super(ParallelDecoderLayer, self).__init__()
        if output_layer_init_method is None:
            output_layer_init_method = init_method

        self.input_layernorm = LayerNorm(hidden_size, eps=layernorm_epsilon)

        self.self_attention = ParallelSelfAttention(
            hidden_size,
            num_attention_heads,
            attention_dropout_prob,
            output_dropout_prob,
            init_method,
            output_layer_init_method=output_layer_init_method)

        self.post_self_layernorm = LayerNorm(hidden_size, eps=layernorm_epsilon)

        self.cross_attention = ParallelCrossAttention(
            hidden_size,
            num_attention_heads,
            attention_dropout_prob,
            output_dropout_prob,
            init_method,
            output_layer_init_method=output_layer_init_method
        )

        self.post_attention_layernorm = LayerNorm(hidden_size, eps=layernorm_epsilon)

        self.mlp = ParallelMLP(
            hidden_size,
            output_dropout_prob,
            init_method,
            output_layer_init_method=output_layer_init_method)

    def forward(self, hidden_states, encoder_states, ltor_mask, cross_mask=None):
        layernorm_output = self.input_layernorm(hidden_states)
       
        self_attention_output = self.self_attention(layernorm_output, ltor_mask)
        
        self_layernorm_input = hidden_states + self_attention_output
        
        self_layernorm_output = self.post_self_layernorm(self_layernorm_input)
      
        attention_output = self.cross_attention(self_layernorm_output, encoder_states, cross_mask)
      
        layernorm_input = self_layernorm_input + attention_output
        
        layernorm_output = self.post_attention_layernorm(layernorm_input)
       
        mlp_output = self.mlp(layernorm_output)
       
        output = layernorm_input + mlp_output
        return output


class ParallelTransformerLayer(flow.nn.Module):
    def __init__(self,
                 hidden_size,
                 num_attention_heads,
                 attention_dropout_prob,
                 output_dropout_prob,
                 layernorm_epsilon,
                 init_method,
                 output_layer_init_method=None,
                 relative_encoding=False,
                 performer=False,
                 attention_scale=1.0):
        super(ParallelTransformerLayer, self).__init__()
    
        if output_layer_init_method is None:
            output_layer_init_method = init_method

        self.input_layernorm = LayerNorm(hidden_size, eps=layernorm_epsilon)

        self.attention = ParallelSelfAttention(
            hidden_size,
            num_attention_heads,
            attention_dropout_prob,
            output_dropout_prob,
            init_method,
            output_layer_init_method=output_layer_init_method,
            relative_encoding=relative_encoding,
            performer=performer,
            attention_scale=attention_scale)

        self.post_attention_layernorm = LayerNorm(hidden_size,
                                                  eps=layernorm_epsilon)

        self.mlp = ParallelMLP(
            hidden_size,
            output_dropout_prob,
            init_method,
            output_layer_init_method=output_layer_init_method)

    def forward(self, hidden_states, ltor_mask, position_embeddings=None, r_w_bias=None, r_r_bias=None, mem=None):
       
        layernorm_output = self.input_layernorm(hidden_states)
        mem = self.input_layernorm(mem) if mem is not None else None

        attention_output = self.attention(layernorm_output, ltor_mask, position_embeddings, r_w_bias, r_r_bias, mem)
        
        
        layernorm_input = hidden_states + attention_output
        
        
        layernorm_output = self.post_attention_layernorm(layernorm_input)
       
        mlp_output = self.mlp(layernorm_output)
       
        output = layernorm_input + mlp_output

        return output


def unscaled_init_method(sigma):
    

    def init_(tensor):
        return flow.nn.init.normal_(tensor, mean=0.0, std=sigma)

    return init_


def scaled_init_method(sigma, num_layers):
  
    std = sigma / math.sqrt(2.0 * num_layers)

    def init_(tensor):
        return flow.nn.init.normal_(tensor, mean=0.0, std=std)

    return init_


class GPT2ParallelTransformer(flow.nn.Module):

    def __init__(self,
                 num_layers,
                 hidden_size,
                 num_attention_heads,
                 max_sequence_length,
                 max_memory_length,
                 embedding_dropout_prob,
                 attention_dropout_prob,
                 output_dropout_prob,
                 checkpoint_activations,
                 checkpoint_num_layers=1,
                 layernorm_epsilon=1.0e-5,
                 init_method_std=0.02,
                 use_scaled_init_for_output_weights=True,
                 relative_encoding=False,
                 block_position_encoding=False,
                 performer=False,
                 use_decoder_layer=False,
                 attention_scale=1.0,
                 ):
        super(GPT2ParallelTransformer, self).__init__()
        self.hidden_size = hidden_size
      
        self.checkpoint_activations = checkpoint_activations
        self.checkpoint_num_layers = checkpoint_num_layers
        self.max_memory_length = max_memory_length
        self.performer = performer
        self.use_decoder_layer = use_decoder_layer
        assert not (performer and relative_encoding)

        output_layer_init_method = None
        
        if use_scaled_init_for_output_weights:
            output_layer_init_method = scaled_init_method(init_method_std,
                                                          num_layers)
       
        self.embedding_dropout = flow.nn.Dropout(embedding_dropout_prob)
        self.relative_encoding = relative_encoding
        self.block_position_encoding = block_position_encoding
        #False
        if relative_encoding:
           
            self.position_embeddings = PositionalEmbedding(hidden_size)
          
            world_size = get_model_parallel_world_size()
            self.hidden_size_per_attention_head = divide(hidden_size,
                                                         num_attention_heads)
            self.num_attention_heads_per_partition = divide(num_attention_heads,
                                                            world_size)
            self.r_w_bias = flow.nn.Parameter(
                flow.Tensor(self.num_attention_heads_per_partition, self.hidden_size_per_attention_head))
            self.r_w_bias.model_parallel = True
            self.r_r_bias = flow.nn.Parameter(
                flow.Tensor(self.num_attention_heads_per_partition, self.hidden_size_per_attention_head))
            self.r_r_bias.model_parallel = True
        
            with flow.no_grad():
                self.r_w_bias.zero_()
                self.r_r_bias.zero_()
        else:
            #True
            if block_position_encoding:
                self.position_embeddings = flow.nn.Embedding(max_sequence_length + 1, hidden_size)
                self.block_position_embeddings = flow.nn.Embedding(max_sequence_length + 1, hidden_size)
                flow.nn.init.normal_(self.block_position_embeddings.weight, mean=0.0, std=init_method_std)
            else:
                self.position_embeddings = flow.nn.Embedding(max_sequence_length, hidden_size)
            flow.nn.init.normal_(self.position_embeddings.weight, mean=0.0, std=init_method_std)

        def get_layer():
            return ParallelTransformerLayer(
                hidden_size,
                num_attention_heads,
                attention_dropout_prob,
                output_dropout_prob,
                layernorm_epsilon,
                unscaled_init_method(init_method_std),
                output_layer_init_method=output_layer_init_method,
                relative_encoding=relative_encoding,
                performer=performer,
                attention_scale=attention_scale)

        self.layers = flow.nn.ModuleList(
            [get_layer() for _ in range(num_layers)])

        self.final_layernorm = LayerNorm(hidden_size, eps=layernorm_epsilon)

        self.register_buffer("ids", flow._C.arange(332, dtype=flow.int64).view(1, -1))
        
    def forward(self, hidden_states, position_ids, attention_mask):     
        batch_size, query_length = hidden_states.size()[:2]
        memory_length = 0
        key_length = query_length + memory_length
        
        is_scalar = flow.numel(attention_mask) == 1
        is_sep = is_scalar or flow.numel(attention_mask) == batch_size

        #False
        if self.performer:
            assert is_scalar, 'attention_mask should be a scalar to indicate the seperation position.'
            assert memory_length == 0, 'Do not support transformer-xl.'
        
        #True
        if is_sep:
            sep = attention_mask.item() if is_scalar else attention_mask
            
            def build_mask_matrix(seq_length, sep, memory_length=0):
                if hidden_states.is_consistent:
                    m = flow.ones((1, seq_length, seq_length), placement=hidden_states.placement, sbp=flow.sbp.broadcast, dtype=hidden_states.dtype)
                else:
                    m = hidden_states.new_ones((1, seq_length, seq_length))
                m = flow.tril(m)
                
                #False
                if is_scalar:
                    m[0, :, :sep] = 1
                else:
                    m = m.expand(batch_size, -1, -1)
                    # ids = flow._C.arange(seq_length, device=sep.device, dtype=sep.dtype).view(1, -1)
                    ids = self.ids
                    mask = ids < sep.view(-1, 1)
                    
                    #expand_as 不支持
                    m = m.masked_fill(mask.unsqueeze(1).expand_as(m), 1)
                   
                m = m.unsqueeze(1)
                return m
            
            #True
            if not self.performer:
                attention_mask = build_mask_matrix(query_length, sep, memory_length=memory_length)
            
        else:
            attention_mask = attention_mask[:, :, :, -query_length - memory_length:]
        
        #False
        if self.relative_encoding:
            position_sequence = flow._C.arange(key_length - 1, -1, -1.0, device=hidden_states.device,
                                             dtype=hidden_states.dtype)
            position_embeddings = self.position_embeddings(position_sequence)

            position_embeddings = self.embedding_dropout(position_embeddings)
        else:
            #true
            if self.block_position_encoding:
                position_ids, block_position_ids = position_ids[:, 0], position_ids[:, 1]
            position_embeddings = self.position_embeddings(position_ids)
            hidden_states = hidden_states + position_embeddings
            #true
            if self.block_position_encoding:
                block_position_embeddings = self.block_position_embeddings(block_position_ids)
                hidden_states = hidden_states + block_position_embeddings
        hidden_states = self.embedding_dropout(hidden_states)
   
        mem_layers = []

        def custom(start, end):
            def custom_forward(*inputs):
                layers_ = self.layers[start:end]
                x_, inputs = inputs[0], inputs[1:]
                if self.relative_encoding:
                    inputs, mems_ = inputs[:4], inputs[4:]
                else:
                    inputs, mems_ = inputs[:1], inputs[1:]
                for i, layer in enumerate(layers_):
                    mem_i_ = mems_[i] if mems_ else None
                    x_ = layer(x_, *inputs)
                return x_

            return custom_forward
    
        
        for i, layer in enumerate(self.layers):
            args = [hidden_states, attention_mask]
            if self.relative_encoding:
                args += [position_embeddings, self.r_w_bias, self.r_r_bias]
            mem_i = None
            hidden_states = layer(*args)
        output = self.final_layernorm(hidden_states)
        
        #False
        return (output, mem_layers)

    def update_mems(self, hiddens, mems):
        memory_length = 0
        query_length = hiddens[0].size(1)
        new_memory_length = memory_length + query_length
        new_mems = []
       
        for i in range(len(hiddens)):
            if new_memory_length <= query_length:
                new_mems.append(hiddens[i][:, -new_memory_length:])
            else:
                new_mems.append(flow.cat((mems[i][:, -new_memory_length + query_length:], hiddens[i]), dim=1))
        return new_mems


class BertParallelSelfAttention(flow.nn.Module):
  
    def __init__(self, hidden_size, num_attention_heads,
                 dropout_prob, output_parallel=False,
                 init_method=init.xavier_normal_):
        super(BertParallelSelfAttention, self).__init__()
       
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.dropout_prob = dropout_prob
        self.output_parallel = output_parallel
       
        world_size = get_model_parallel_world_size()
        self.hidden_size_per_partition = divide(hidden_size, world_size)
        self.hidden_size_per_attention_head = divide(hidden_size,
                                                     num_attention_heads)
        self.num_attention_heads_per_partition = divide(num_attention_heads,
                                                        world_size)

        self.query_key_value = ColumnParallelLinear(hidden_size, 3 * hidden_size,
                                                    stride=3,
                                                    gather_output=False,
                                                    init_method=init_method)

        self.dropout = flow.nn.Dropout(dropout_prob)

    def _transpose_for_scores(self, tensor):
    
        new_tensor_shape = tensor.size()[:-1] + \
                           (self.num_attention_heads_per_partition,
                            self.hidden_size_per_attention_head)
        tensor = tensor.view(*new_tensor_shape)
        return tensor.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask):

        mixed_x_layer = self.query_key_value(hidden_states)
        (mixed_query_layer,
         mixed_key_layer,
         mixed_value_layer) = split_tensor_along_last_dim(mixed_x_layer, 3)

        query_layer = self._transpose_for_scores(mixed_query_layer)
        key_layer = self._transpose_for_scores(mixed_key_layer)
        value_layer = self._transpose_for_scores(mixed_value_layer)

        norm_factor = math.sqrt(math.sqrt(self.hidden_size_per_attention_head))
        attention_scores = flow.matmul(query_layer / norm_factor,
                                        key_layer.transpose(-1, -2) / norm_factor)
        attention_scores += attention_mask

        attention_probs = flow.nn.Softmax(dim=-1)(attention_scores)
        attention_probs = self.dropout(attention_probs)

        context_layer = flow.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        # new_context_layer_shape = context_layer.size()[:-2] + \
        #                           (self.hidden_size_per_partition,)
        context_layer_size = context_layer.size()
        new_context_layer_shape = [context_layer_size[0], context_layer_size[1],self.hidden_size_per_partition]

        context_layer = context_layer.view(*new_context_layer_shape)

        if self.output_parallel:
            output = context_layer

        return output


class BertParallelTransformerOutput(flow.nn.Module):

    def __init__(self, input_size, output_size, dropout_prob,
                 layernorm_epsilon=1.0e-12, input_is_parallel=False,
                 init_method=init.xavier_normal_):
        super(BertParallelTransformerOutput, self).__init__()
       
        self.dense = RowParallelLinear(input_size,
                                       output_size,
                                       input_is_parallel=input_is_parallel,
                                       init_method=init_method)
        self.dropout = flow.nn.Dropout(dropout_prob)
        self.layernorm = LayerNorm(output_size, eps=layernorm_epsilon)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        layernorm_input = hidden_states + input_tensor
        hidden_states = self.layernorm(layernorm_input)
        return hidden_states


class BertParallelTransformerLayer(flow.nn.Module):

    def __init__(self,
                 hidden_size,
                 intermediate_size,
                 num_attention_heads,
                 attention_dropout_prob,
                 output_dropout_prob,
                 intermediate_activation_fn,
                 layernorm_epsilon,
                 init_method=init.xavier_normal_):
        super(BertParallelTransformerLayer, self).__init__()

        self.attention = BertParallelSelfAttention(hidden_size,
                                                   num_attention_heads,
                                                   attention_dropout_prob,
                                                   output_parallel=True,
                                                   init_method=init_method)
    
        self.self_output = BertParallelTransformerOutput(
            hidden_size, hidden_size, output_dropout_prob,
            layernorm_epsilon=layernorm_epsilon,
            input_is_parallel=True,
            init_method=init_method)
        
        self.intermediate = ColumnParallelLinear(hidden_size, intermediate_size,
                                                 gather_output=False,
                                                 init_method=init_method)
        self.intermediate_activation_fn = intermediate_activation_fn
        
        self.output = BertParallelTransformerOutput(
            intermediate_size, hidden_size, output_dropout_prob,
            layernorm_epsilon=layernorm_epsilon,
            input_is_parallel=True,
            init_method=init_method)

    def forward(self, hidden_states, attention_mask):
      
        attention_output_parallel = self.attention(hidden_states,
                                                   attention_mask)
    
        attention_self_output = self.self_output(attention_output_parallel,
                                                 hidden_states)
     
        intermediate_output_parallel = self.intermediate(attention_self_output)
        intermediate_output_parallel = self.intermediate_activation_fn(
            intermediate_output_parallel)
      
        layer_output = self.output(intermediate_output_parallel,
                                   attention_self_output)

        return layer_output
