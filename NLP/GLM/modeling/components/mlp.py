import oneflow as flow 
import oneflow.nn.init as init
import oneflow.nn.functional as F
from oneflow.nn.parameter import Parameter

from .utils import divide, _initialize_affine_weight

class ColumnParallelLinear(flow.nn.Module):
    def __init__(self, input_size, output_size, bias=True, gather_output=True,
                 init_method=init.xavier_normal_, stride=1,
                 keep_master_weight_for_test=False, 
                 if_use_gelu=False, 
                 if_use_dropout=False, 
                 dropout_rate=0.0
                 ):
        super(ColumnParallelLinear, self).__init__()

        self.if_use_gelu = if_use_gelu
        self.if_use_dropout = if_use_dropout
        self.dropout_rate = dropout_rate

        self.input_size = input_size
        self.output_size = output_size
        self.gather_output = gather_output
        
        world_size = 1
        # world_size = get_model_parallel_world_size()

        self.output_size_per_partition = divide(output_size, world_size)

        self.weight = Parameter(flow.Tensor(self.output_size_per_partition,
                                             self.input_size))
        # self.weight.model_parallel = True
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

        # previous 
        # output_parallel = F.linear(input_parallel, self.weight, self.bias)

        # Fused!
        output_parallel = F.linear(input_parallel, self.weight)
        if self.if_use_gelu: 
            return flow._C.fused_bias_add_gelu(output_parallel, self.bias, axis=2)
        elif self.dropout_rate - 0.0 < 1e-9:
            return flow._C.bias_add(output_parallel, self.bias, axis=2)
        elif self.if_use_dropout: 
            return flow._C.fused_bias_add_dropout(output_parallel, self.bias, p=self.dropout_rate, axis=2)
        else: 
            # Do nothing
            return flow._C.bias_add(output_parallel, self.bias, axis=2)


class RowParallelLinear(flow.nn.Module):
    def __init__(self, input_size, output_size, bias=True,
                 input_is_parallel=False,
                 init_method=init.xavier_normal_, stride=1,
                 keep_master_weight_for_test=False, 
                 if_use_gelu=False, 
                 if_use_dropout=False, 
                 dropout_rate=0.0):
        super(RowParallelLinear, self).__init__()

        self.if_use_gelu = if_use_gelu 
        self.if_use_dropout = if_use_dropout
        self.dropout_rate = dropout_rate

        self.input_size = input_size
        self.output_size = output_size
        self.input_is_parallel = input_is_parallel

        world_size = 1
        # world_size = get_model_parallel_world_size()

        self.input_size_per_partition = divide(input_size, world_size)

    
        self.weight = Parameter(flow.Tensor(self.output_size,
                                             self.input_size_per_partition))
        # self.weight.model_parallel = True

        if bias:
            self.bias = Parameter(flow.Tensor(self.output_size))
        else:
            self.register_parameter('bias', None)
        self.master_weight = _initialize_affine_weight(
            self.weight, self.output_size, self.input_size,
            self.input_size_per_partition, 1, init_method,
            stride=stride, return_master_weight=keep_master_weight_for_test)

    def forward(self, input_):
        
        # Fused!
        output_parallel = F.linear(input_, self.weight)
        if self.if_use_gelu: 
            return flow._C.fused_bias_add_gelu(output_parallel, self.bias, axis=2)
        elif self.dropout_rate - 0.0 < 1e-9:
            return flow._C.bias_add(output_parallel, self.bias, axis=2)
        elif self.if_use_dropout: 
            return flow._C.fused_bias_add_dropout(output_parallel, self.bias, p=self.dropout_rate, axis=2)
        else: 
            # Do nothing
            return flow._C.bias_add(output_parallel, self.bias, axis=2)


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