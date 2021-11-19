import oneflow as flow
import oneflow.nn.init as init
from  oneflow.nn import LayerNorm
from .attention import ParallelSelfAttention
from .mlp import ParallelMLP

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