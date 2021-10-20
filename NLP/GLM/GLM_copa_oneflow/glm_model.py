from line_profiler import LineProfiler
import oneflow as flow
import oneflow.nn.functional as F
import oneflow.nn.init as init
from oneflow.nn.parameter import Parameter
from  oneflow.nn import LayerNorm
import oneflow

import math
import deepspeed
import time

def init_method_normal(std=0.02):
    def init_(tensor):
        return flow.nn.init.normal_(tensor, mean=0.0, std=std)
    return init_

def ensure_divisibility(numerator, denominator):
    assert numerator % denominator == 0, '{} is not divisible by {}'.format(
        numerator, denominator)

def divide(numerator, denominator):
    ensure_divisibility(numerator, denominator)
    return numerator // denominator

class VocabUtility:

    @staticmethod
    def vocab_range_from_per_partition_vocab_size(per_partition_vocab_size,
                                                  rank, world_size):
        index_f = rank * per_partition_vocab_size
        index_l = index_f + per_partition_vocab_size
        return index_f, index_l

    @staticmethod
    def vocab_range_from_global_vocab_size(global_vocab_size, rank, world_size):
        per_partition_vocab_size = divide(global_vocab_size, world_size)
        return VocabUtility.vocab_range_from_per_partition_vocab_size(
            per_partition_vocab_size, rank, world_size)

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
    master_weight = flow.empty(output_size, input_size,
                                dtype=weight.dtype,
                                requires_grad=False)
    init_method(master_weight)

    per_partition_per_stride_size = divide(per_partition_size, stride)
    weight_list = flow.split(master_weight, per_partition_per_stride_size,
                              dim=partition_dim)
    rank = get_model_parallel_rank()
   
    my_weight_list = weight_list[rank::world_size]

    with flow.no_grad():
        flow.cat(my_weight_list, dim=partition_dim, out=weight)
    if return_master_weight:
        return master_weight
    return None

class VocabParallelEmbedding(flow.nn.Module):
   
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

        self.weight = Parameter(flow.Tensor(self.num_embeddings_per_partition,
                                             self.embedding_dim))
        # self.weight.model_parallel = True
        _initialize_affine_weight(
            self.weight, self.num_embeddings, self.embedding_dim,
            self.num_embeddings_per_partition, 0, init_method)

    def forward(self, input_):
        ##不支持或操作符
        input_mask = (input_ < self.vocab_start_index) | \
                     (input_ >= self.vocab_end_index)
        oneflow._oneflow_internal.profiler.RangePush('VocabParallelEmbedding-1')
        # input_mask = flow._C.logical_or(input_ < self.vocab_start_index, input_ >= self.vocab_end_index)
        oneflow._oneflow_internal.profiler.RangePop()
        oneflow._oneflow_internal.profiler.RangePush('VocabParallelEmbedding-2')
        masked_input = (input_.clone() - self.vocab_start_index).to(flow.int)
        oneflow._oneflow_internal.profiler.RangePop()
        
        #不支持切片索引
        oneflow._oneflow_internal.profiler.RangePush('VocabParallelEmbedding-3')
        masked_input[input_mask] = 0
        oneflow._oneflow_internal.profiler.RangePop()
        oneflow._oneflow_internal.profiler.RangePush('VocabParallelEmbedding-4')
        output_parallel = F.embedding(masked_input, 
                                      self.weight,
                                      self.padding_idx, 
                                      self.max_norm,
                                      None,               #self.norm_type, 暂时变为none
                                      self.scale_grad_by_freq,
                                      self.sparse)
        oneflow._oneflow_internal.profiler.RangePop()
        oneflow._oneflow_internal.profiler.RangePush('VocabParallelEmbedding-5')                              
        #不支持切片索引
        # output_parallel[input_mask, :] = 0.0
        output_parallel[input_mask[...,None].expand(*output_parallel.shape).to(flow.int8)] = 0
   
        #output = reduce_from_model_parallel_region(output_parallel)
        output = output_parallel
        oneflow._oneflow_internal.profiler.RangePop()
        return output


def scaled_init_method(sigma, num_layers):
  
    std = sigma / math.sqrt(2.0 * num_layers)

    def init_(tensor):
        return flow.nn.init.normal_(tensor, mean=0.0, std=std)

    return init_


def split_tensor_along_last_dim(tensor, num_partitions,
                                contiguous_split_chunks=False):
    last_dim = tensor.dim() - 1
    last_dim_size = divide(tensor.size()[last_dim], num_partitions)
    tensor_list = flow.split(tensor, last_dim_size, dim=last_dim)
    if contiguous_split_chunks:
        return tuple(chunk.contiguous() for chunk in tensor_list)

    return tensor_list


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

        self.dense = RowParallelLinear(hidden_size,
                                       hidden_size,
                                       input_is_parallel=True,
                                       init_method=output_layer_init_method)
        self.output_dropout = flow.nn.Dropout(output_dropout_prob)

        if deepspeed.checkpointing.is_configured():
            global get_cuda_rng_tracker, checkpoint
            get_cuda_rng_tracker = deepspeed.checkpointing.get_cuda_rng_tracker
            checkpoint = deepspeed.checkpointing.checkpoint

    def _transpose_for_scores(self, tensor):
        
        #不支持
        # new_tensor_shape = tensor.size()[:-1] + \
        #                    (self.num_attention_heads_per_partition,
        #                     self.hidden_size_per_attention_head)
        new_tensor_shape = [*tensor.size()[:-1],self.num_attention_heads_per_partition,self.hidden_size_per_attention_head]
       
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

        #with get_cuda_rng_tracker().fork():
        attention_probs = self.attention_dropout(attention_probs)
    
        context_layer = flow.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        
        # new_context_layer_shape = context_layer.size()[:-2] + \
        #                           (self.hidden_size_per_partition,)
        new_context_layer_shape = [*context_layer.size()[:-2],self.hidden_size_per_partition]
        context_layer = context_layer.reshape(*new_context_layer_shape)
        
        output = self.dense(context_layer)
        output = self.output_dropout(output)

        return output


class ColumnParallelLinear(flow.nn.Module):
    def __init__(self, input_size, output_size, bias=True, gather_output=True,
                 init_method=init.xavier_normal_, stride=1,
                 keep_master_weight_for_test=False):
        super(ColumnParallelLinear, self).__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.gather_output = gather_output
        
        world_size = 1
    
        self.output_size_per_partition = divide(output_size, world_size)

        self.weight = Parameter(flow.Tensor(self.output_size_per_partition,
                                             self.input_size))
    
        if bias:
            self.bias = Parameter(flow.Tensor(self.output_size_per_partition))
        else:
            self.register_parameter('bias', None)

        self.master_weight = _initialize_affine_weight(
            self.weight, self.output_size, self.input_size,
            self.output_size_per_partition, 0, init_method,
            stride=stride, return_master_weight=keep_master_weight_for_test)

    def forward(self, input_):

        input_parallel = input_
        output_parallel = F.linear(input_parallel, self.weight, self.bias)
        return output_parallel


class RowParallelLinear(flow.nn.Module):
    def __init__(self, input_size, output_size, bias=True,
                 input_is_parallel=False,
                 init_method=init.xavier_normal_, stride=1,
                 keep_master_weight_for_test=False):
        super(RowParallelLinear, self).__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.input_is_parallel = input_is_parallel

        world_size = 1
       
        self.input_size_per_partition = divide(input_size, world_size)

    
        self.weight = Parameter(flow.Tensor(self.output_size,
                                             self.input_size_per_partition))

        if bias:
            self.bias = Parameter(flow.Tensor(self.output_size))
        else:
            self.register_parameter('bias', None)
        self.master_weight = _initialize_affine_weight(
            self.weight, self.output_size, self.input_size,
            self.input_size_per_partition, 1, init_method,
            stride=stride, return_master_weight=keep_master_weight_for_test)

    def forward(self, input_):
      
        output_parallel = F.linear(input_, self.weight)
      
        if self.bias is not None:
            output = output_parallel + self.bias
        else:
            output = output_parallel
        return output

def gelu_impl(x):
    return 0.5 * x * (1.0 + flow.tanh(0.7978845608028654 * x *
                                       (1.0 + 0.044715 * x * x)))
def gelu(x):
    return gelu_impl(x)

class ParallelMLP(flow.nn.Module):

    def __init__(self, hidden_size, output_dropout_prob, init_method,
                 output_layer_init_method=None):
        super(ParallelMLP, self).__init__()
     
        if output_layer_init_method is None:
            output_layer_init_method = init_method
       
        self.dense_h_to_4h = ColumnParallelLinear(hidden_size, 4 * hidden_size,
                                                  gather_output=False,
                                                  init_method=init_method)
       
        self.dense_4h_to_h = RowParallelLinear(
            4 * hidden_size,
            hidden_size,
            input_is_parallel=True,
            init_method=output_layer_init_method)
        self.dropout = flow.nn.Dropout(output_dropout_prob)

    def forward(self, hidden_states):
        intermediate_parallel = self.dense_h_to_4h(hidden_states)
        intermediate_parallel = gelu(intermediate_parallel)

        output = self.dense_4h_to_h(intermediate_parallel)
        output = self.dropout(output)
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

        #True
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
            #False
            if use_decoder_layer:
                return ParallelDecoderLayer(
                    hidden_size,
                    num_attention_heads,
                    attention_dropout_prob,
                    output_dropout_prob,
                    layernorm_epsilon,
                    unscaled_init_method(init_method_std),
                    output_layer_init_method=output_layer_init_method
                )
            else:
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
        
        #False
        if deepspeed.checkpointing.is_configured():
            global get_cuda_rng_tracker, checkpoint
            get_cuda_rng_tracker = deepspeed.checkpointing.get_cuda_rng_tracker
            checkpoint = deepspeed.checkpointing.checkpoint

    def forward(self, hidden_states, position_ids, attention_mask, memory_states=None, encoder_states=None,
                return_memory=False, detach_memory=True):     
        flow.ones((16,16,16,16))
        flow.zeros((16,16,16,16))
        batch_size, query_length = hidden_states.size()[:2]
        memory_length = memory_states[0].size(1) if memory_states else 0
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
                m = hidden_states.new_ones((1, seq_length, seq_length))
                m = flow.tril(m)
                
                #False
                if is_scalar:
                    m[0, :, :sep] = 1
                else:
                    m = m.expand(batch_size, -1, -1)
                    ids = flow.arange(seq_length, device=sep.device, dtype=sep.dtype).view(1, -1)
                    mask = ids < sep.view(-1, 1)
                    
                    #expand_as 不支持
                    m = m.masked_fill(mask.unsqueeze(1).expand_as(m), 1)
                   
                #False
                if memory_length > 0:
                    m = m.expand(batch_size, -1, -1)
                    m = flow.cat((hidden_states.new_ones((batch_size, seq_length, memory_length)), m), dim=2)
                m = m.unsqueeze(1)
                return m
            
            #True
            if not self.performer:
                attention_mask = build_mask_matrix(query_length, sep, memory_length=memory_length)
            
        else:
            attention_mask = attention_mask[:, :, :, -query_length - memory_length:]
        
        #False
        if self.relative_encoding:
            position_sequence = flow.arange(key_length - 1, -1, -1.0, device=hidden_states.device,
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

        def check_detach(_hidden_states):
            if detach_memory:
                return _hidden_states.detach()
            return _hidden_states
        
        #False
        if self.max_memory_length > 0 or return_memory:
            mem_layers = [check_detach(hidden_states)]
        else:
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
                    x_ = layer(x_, *inputs, mem=mem_i_)
                    if self.max_memory_length > 0 or return_memory:
                        mem_layers.append(check_detach(x_))
                return x_

            return custom_forward
    
        #True
        if self.checkpoint_activations:
            l = 0
            num_layers = len(self.layers)
            chunk_length = self.checkpoint_num_layers
            
            while l < num_layers:
                args = [hidden_states, attention_mask] if not self.use_decoder_layer else [hidden_states,
                                                                                           encoder_states,
                                                                                           attention_mask]
                #False
                if self.relative_encoding:
                    args += [position_embeddings, self.r_w_bias, self.r_r_bias]
                #False
                if memory_states:
                    args += memory_states[l: l + chunk_length]

                #hidden_states = checkpoint(custom(l, l + chunk_length), *args)
                hidden_states = custom(l, l + chunk_length)(*args)
                l += chunk_length
        else:
            for i, layer in enumerate(self.layers):
                args = [hidden_states, attention_mask] if not self.use_decoder_layer else [hidden_states,
                                                                                           encoder_states,
                                                                                           attention_mask]
                if self.relative_encoding:
                    args += [position_embeddings, self.r_w_bias, self.r_r_bias]
                mem_i = memory_states[i] if memory_states else None
                hidden_states = layer(*args, mem=mem_i)
                if self.max_memory_length > 0 or return_memory:
                    mem_layers.append(check_detach(hidden_states))
        output = self.final_layernorm(hidden_states)

        #False
        if self.max_memory_length > 0 or return_memory:
            mem_layers = self.update_mems(mem_layers, memory_states, return_memory=return_memory)
        return (output, mem_layers)

    def update_mems(self, hiddens, mems, return_memory=False):
        memory_length = mems[0].size(1) if mems else 0
        query_length = hiddens[0].size(1)
        new_memory_length = memory_length + query_length
        if not return_memory:
            new_memory_length = min(self.max_memory_length, new_memory_length)
        new_mems = []
       
        for i in range(len(hiddens)):
            if new_memory_length <= query_length:
                new_mems.append(hiddens[i][:, -new_memory_length:])
            else:
                new_mems.append(flow.cat((mems[i][:, -new_memory_length + query_length:], hiddens[i]), dim=1))
        return new_mems


class GLMModel(flow.nn.Module):
    def __init__(self,
                 num_layers,
                 vocab_size,
                 hidden_size,
                 num_attention_heads,
                 embedding_dropout_prob,
                 attention_dropout_prob,
                 output_dropout_prob,
                 max_sequence_length,
                 max_memory_length,
                 checkpoint_activations,
                 checkpoint_num_layers=1,
                 parallel_output=True,
                 relative_encoding=False,
                 block_position_encoding=False,
                 output_predict=True,
                 spell_length=None,
                 spell_func='lstm',
                 attention_scale=1.0,
                 ):

        super(GLMModel, self).__init__()

        self.parallel_output = parallel_output
        self.output_predict = output_predict
        self.hidden_size = hidden_size

        init_method = init_method_normal(std=0.02)

        # Word embeddings (parallel).
        self.word_embeddings = VocabParallelEmbedding(
            vocab_size, hidden_size, init_method=init_method)

        # Transformer
        self.transformer = GPT2ParallelTransformer(num_layers,
                                                   hidden_size,
                                                   num_attention_heads,
                                                   max_sequence_length,
                                                       max_memory_length,
                                                       embedding_dropout_prob,
                                                       attention_dropout_prob,
                                                       output_dropout_prob,
                                                       checkpoint_activations,
                                                       checkpoint_num_layers,
                                                       attention_scale=attention_scale,
                                                       relative_encoding=relative_encoding,
                                                       block_position_encoding=block_position_encoding)
        #False
        if spell_length is not None:
            self.prompt_spell = PromptSpell(spell_length, self.hidden_size, spell_func)

    def freeze_transformer(self, tune_prefix_layers=None):
        log_str = "Freeze transformer"
        self.word_embeddings.requires_grad_(False)
        self.transformer.requires_grad_(False)
        if tune_prefix_layers is not None:
            log_str += f" tune {tune_prefix_layers} prefix layers"
            for i in range(tune_prefix_layers):
                self.transformer.layers[i].requires_grad_(True)
        print_rank_0(log_str)

    def forward(self, input_ids, position_ids, attention_mask, *mems, return_memory=False, detach_memory=True,
                prompt_pos=None):
        # Embeddings.
        batch_size = input_ids.size(0)
        oneflow._oneflow_internal.profiler.RangePush('embeddings')
        words_embeddings = self.word_embeddings(input_ids)
        oneflow._oneflow_internal.profiler.RangePop()
        embeddings = words_embeddings
        #False
        if prompt_pos is not None:
            embeddings = embeddings.clone()
            prompt_embeds = self.prompt_spell()
            batch_index = flow.arange(batch_size, device=input_ids.device).unsqueeze(1)
            embeddings[batch_index, prompt_pos] = prompt_embeds
        # Transformer.
        oneflow._oneflow_internal.profiler.RangePush('transformer')
        transformer_output = self.transformer(embeddings, position_ids, attention_mask, mems,
                                              return_memory=return_memory, detach_memory=detach_memory)
        oneflow._oneflow_internal.profiler.RangePop()
        logits, hidden_layers = transformer_output
        outputs = hidden_layers
        
        #True
        if self.output_predict:

            # logits_parallel = mpu.copy_to_model_parallel_region(
            #     logits)
            logits_parallel = logits
            logits_parallel = F.linear(logits_parallel, self.word_embeddings.weight)
            
            if self.parallel_output:
                return (logits_parallel, *outputs)
            
            return (logits_parallel,*outputs)
            #return (mpu.gather_from_model_parallel_region(logits_parallel), *outputs)
        else:
            return (logits, *outputs)


class GLMForMultiTokenCloze(flow.nn.Module):

    def __init__(self, language_model: GLMModel, take_softmax=True, length_penalty=0.0):
        super(GLMForMultiTokenCloze, self).__init__()
        self.model = language_model
        self.take_softmax = take_softmax
        self.length_penalty = length_penalty

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        sd = self.model.state_dict(destination, prefix, keep_vars)
        return sd

    def load_state_dict(self, state_dict, strict=True):
        return self.model.load_state_dict(state_dict, strict=strict)

    def named_parameters(self, prefix: str = '', recurse: bool = True):
        return self.model.named_parameters(prefix=prefix, recurse=recurse)

    def forward(self, input_ids, position_ids, attention_mask, target_ids=None, logit_mask=None, prompt_pos=None):
        
        #False
        if target_ids is None:
            return self.model(input_ids, position_ids, attention_mask)
        num_choices = None
        #True
        if len(input_ids.shape) == 3:
            batch_size, num_choices = input_ids.shape[:2]
            input_ids = input_ids.reshape(-1, input_ids.size(-1))
            attention_mask = attention_mask.reshape(-1, *attention_mask.size()[2:])
            position_ids = position_ids.reshape(-1, *position_ids.size()[2:])
            target_ids = target_ids.reshape(-1, target_ids.size(-1))
            logit_mask = logit_mask.reshape(-1, logit_mask.size(-1))
            #False
            if prompt_pos is not None:
                prompt_pos = prompt_pos.reshape(-1, prompt_pos.size(-1))
        
        outputs, *mems = self.model(input_ids, position_ids, attention_mask, prompt_pos=prompt_pos)
        
        #True
        if self.take_softmax:
            oneflow._oneflow_internal.profiler.RangePush('logsoftmax')
            outputs = flow.nn.LogSoftmax(dim=-1)(outputs)
            oneflow._oneflow_internal.profiler.RangePop()

        batch_ids = flow.arange(target_ids.size(0), dtype=flow.long, device=target_ids.device)
        batch_ids = batch_ids.unsqueeze(1).expand_as(target_ids)
        seq_ids = flow.arange(target_ids.size(-1), dtype=flow.long, device=target_ids.device)
        seq_ids = seq_ids.unsqueeze(0).expand_as(target_ids)
        logits = outputs[batch_ids, seq_ids, target_ids]
        logits = (logits * logit_mask).sum(dim=1)
        
        #False
        if self.length_penalty > 0.0:
            logits = logits / logit_mask.sum(dim=1) ** self.length_penalty
        
        #True
        if num_choices is not None:
            logits = logits.view(-1, num_choices)
        return (logits, *mems)

def np_data(shape):
    tensor = flow.ones(shape)
    return tensor

def test():
    def read(path):
        input = []
        with open(path,'r') as f:
            lines = f.readlines()
            for i in lines[0].split():
                input.append(int(i))
        input = flow.Tensor(input).to(flow.int64).cuda()
        return input

    def np_data(shape):
        tensor = flow.ones(shape).to(flow.int64).cuda()
        return tensor

    glmmodel = GLMModel(num_layers=12,
                     vocab_size=30592,
                     hidden_size=768,
                     num_attention_heads=12,
                     embedding_dropout_prob=0.1,
                     attention_dropout_prob=0.1,
                     output_dropout_prob=0.1,
                     max_sequence_length=512,
                     max_memory_length=0,
                     checkpoint_activations=True,
                     checkpoint_num_layers=1,
                     parallel_output=False,
                     relative_encoding=False,
                     block_position_encoding=True,
                     output_predict=True,
                     spell_length=None,
                     spell_func="lstm",
                     attention_scale=1.0)

    model = GLMForMultiTokenCloze(glmmodel, length_penalty=0)
    model = model.cuda()
    model.eval()
    with flow.no_grad():
        inputs_a = [np_data((4,2,256)),
                    np_data((4,2,2,256)),
                    np_data((4,2)),
                    np_data((4,2,256)),
                    np_data((4,2,256))]

        b = time.time()
        for i in range(100):
            oneflow._oneflow_internal.profiler.RangePush('item')
            logits, *mems = model(*inputs_a)
            oneflow._oneflow_internal.profiler.RangePop()
        e = time.time()
        print(e-b)


if __name__ == '__main__':
    # lp = LineProfiler()
    # lp_wrapper = lp(test)
    # lp_wrapper()
    # lp.print_stats()
    test()

    
