import oneflow.experimental as flow
import oneflow.experimental.nn as nn
import math

#Reference: https://github.com/piEsposito/pytorch-lstm-by-hand
class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, batch_size=1,
                    num_layers=1):
        super(LSTM, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.num_layers = num_layers
 
        # Define the LSTM layer
        self.lstm = CustomLSTM(self.input_dim, self.hidden_dim)
 
        # Define the output layer
        self.linear = nn.Linear(self.hidden_dim, output_dim)

    def forward(self, input):
        # Forward pass through LSTM layer
        # shape of lstm_out: [input_size, batch_size, hidden_dim]
        # shape of self.hidden: (a, b), where a and b both 
        # have shape (num_layers, batch_size, hidden_dim).
        lstm_out, _ = self.lstm(input.reshape((input.shape[0], self.batch_size, -1)))
        
        # Only take the output from the final timetep
        # Can pass on the entirety of lstm_out to the next layer if it is a seq2seq prediction
        #NOTE(Xu Zhiqiu) Negative indexing not supported
        output = lstm_out[lstm_out.shape[0] - 1].reshape((self.batch_size, -1))
        y_pred = self.linear(output)
        return y_pred

class CustomLSTM(nn.Module):
    def __init__(self, input_sz, hidden_sz):
        super().__init__()
        self.input_sz = input_sz
        self.hidden_size = hidden_sz
        self.W = nn.Parameter(flow.Tensor(input_sz, hidden_sz * 4))
        self.U = nn.Parameter(flow.Tensor(hidden_sz, hidden_sz * 4))
        self.bias = nn.Parameter(flow.Tensor(hidden_sz * 4))
        self.init_weights()
                
    def init_weights(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)
         
    def forward(self, x, 
                init_states=None):
        """Assumes x is of shape (batch, sequence, feature)"""
        seq_sz, bs, _ = x.size()
        hidden_seq = []
        if init_states is None:
            h_t, c_t = (flow.zeros((bs, self.hidden_size)).to("cuda"), 
                        flow.zeros((bs, self.hidden_size)).to("cuda"))
        else:
            h_t, c_t = init_states
         
        HS = self.hidden_size
        for t in range(seq_sz):
            x_t = x[t, :, :].reshape((x.shape[1], x.shape[2]))
            # batch the computations into a single matrix multiplication
            # NOTE(Xu Zhiqiu): flow does not support view now, use reshape instead
            gates = flow.matmul(x_t, self.W) + flow.matmul(h_t, self.U) + self.bias
            i_t, f_t, g_t, o_t = (
                flow.sigmoid(gates[:, :HS]), # input
                flow.sigmoid(gates[:, HS:HS*2]), # forget
                flow.tanh(gates[:, HS*2:HS*3]),
                flow.sigmoid(gates[:, HS*3:]), # output
            )
            c_t = f_t * c_t + i_t * g_t
            h_t = o_t * flow.tanh(c_t)
            hidden_seq.append(h_t.unsqueeze(0))
        hidden_seq = flow.cat(hidden_seq, dim=0)
        return hidden_seq, (h_t, c_t)

