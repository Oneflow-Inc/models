import math 

import oneflow as flow
import oneflow.nn as nn
import oneflow.nn.functional as F
from  oneflow.nn import LayerNorm

from modeling.components import VocabParallelEmbedding, PositionalEmbedding
from modeling.components import divide, ParallelSelfAttention, ParallelMLP


def init_method_normal(std=0.02):
    def init_(tensor):
        return flow.nn.init.normal_(tensor, mean=0.0, std=std)
    return init_

def unscaled_init_method(sigma):
    def init_(tensor):
        return flow.nn.init.normal_(tensor, mean=0.0, std=sigma)
    return init_


def scaled_init_method(sigma, num_layers):
    std = sigma / math.sqrt(2.0 * num_layers)
    def init_(tensor):
        return flow.nn.init.normal_(tensor, mean=0.0, std=std)
    return init_


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
                 attention_scale=1.0,
                 ):
        super(GPT2ParallelTransformer, self).__init__()
        self.hidden_size = hidden_size
      
        self.checkpoint_activations = checkpoint_activations
        self.checkpoint_num_layers = checkpoint_num_layers
        self.max_memory_length = max_memory_length
        self.performer = performer
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
          
            world_size = 1
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
        
        #True
        if is_sep:
            sep = attention_mask.item() if is_scalar else attention_mask
            
            def build_mask_matrix(seq_length, sep, memory_length=0):
                if hidden_states.is_consistent:
                    m = flow.ones((batch_size, seq_length, seq_length), placement=hidden_states.placement, sbp=flow.sbp.split(0), dtype=hidden_states.dtype)
                else:
                    m = hidden_states.new_ones((1, seq_length, seq_length))
                m = flow.tril(m)
                
                #False
                if is_scalar:
                    m[0, :, :sep] = 1
                else:
                    if not m.is_consistent:
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
    
        
        for i, layer in enumerate(self.layers):
            args = [hidden_states, attention_mask]
            if self.relative_encoding:
                args += [position_embeddings, self.r_w_bias, self.r_r_bias]
            mem_i = None
            hidden_states = layer(*args)
        output = self.final_layernorm(hidden_states)
        
        #False
        return (output, mem_layers)



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
        
        self.num_layers = num_layers
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.embedding_dropout_prob = embedding_dropout_prob
        self.attention_dropout_prob = attention_dropout_prob
        self.output_dropout_prob = output_dropout_prob
        self.max_sequence_length = max_sequence_length
        self.max_memory_length = max_memory_length
        self.checkpoint_activations = checkpoint_activations
        self.checkpoint_num_layers = checkpoint_num_layers
        self.parallel_output = parallel_output
        self.relative_encoding = relative_encoding
        self.block_position_encoding = block_position_encoding
        self.output_predict = output_predict
        self.spell_length = spell_length
        self.spell_func = spell_func
        self.attention_scale = attention_scale


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

    def freeze_transformer(self, tune_prefix_layers=None):
        log_str = "Freeze transformer"
        self.word_embeddings.requires_grad_(False)
        self.transformer.requires_grad_(False)
        if tune_prefix_layers is not None:
            log_str += f" tune {tune_prefix_layers} prefix layers"
            for i in range(tune_prefix_layers):
                self.transformer.layers[i].requires_grad_(True)

    def forward(self, input_ids, position_ids, attention_mask):
        # Embeddings.
        batch_size = input_ids.size(0)
        
        words_embeddings = self.word_embeddings(input_ids)
        embeddings = words_embeddings

        # Transformer.
        transformer_output = self.transformer(embeddings, position_ids, attention_mask)
        logits, hidden_layers = transformer_output
        outputs = hidden_layers
        
        #True
        if self.output_predict:

            logits_parallel = logits
            logits_parallel = F.linear(logits_parallel, self.word_embeddings.weight)
            
            if self.parallel_output:
                return (logits_parallel, *outputs)
            
            return (logits_parallel, *outputs)
        else:
            return (logits, *outputs)