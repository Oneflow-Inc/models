import oneflow as flow
from .mlp import ColumnParallelLinear, RowParallelLinear
import math
from .utils import divide

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
                                       init_method=output_layer_init_method, 
                                       if_use_dropout=True, 
                                       dropout_rate=output_dropout_prob)

    def _transpose_for_scores(self, tensor):
        
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
        
        new_context_layer_shape = [*context_layer.size()[:-2],self.hidden_size_per_partition]
        context_layer = context_layer.reshape(*new_context_layer_shape)
        
        output = self.dense(context_layer) 

        return output
        