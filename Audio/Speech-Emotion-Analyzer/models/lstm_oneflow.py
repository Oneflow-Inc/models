import oneflow as flow
import oneflow.nn as nn
import math


class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, batch_size, num_layers=1):
        super(LSTM, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.num_layers = num_layers
        self.lstm = CustomLSTM(self.input_dim, self.hidden_dim)

    def forward(self, input):
        lstm_out, _ = self.lstm(input.reshape(input.shape[0], self.batch_size, -1))
        output = lstm_out[lstm_out.shape[0] - 1].reshape(self.batch_size, -1)
        return output


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

    def forward(self, x, init_states=None):
        seq_sz, bs, _ = x.size()
        hidden_seq = []
        if init_states is None:
            h_t, c_t = (
                flow.zeros((bs, self.hidden_size)).to("cuda"),
                flow.zeros((bs, self.hidden_size)).to("cuda"),
            )
        else:
            h_t, c_t = init_states

        HS = self.hidden_size
        for t in range(seq_sz):
            x_t = x[t, :, :]
            x_t = x_t.reshape(x.shape[1], x.shape[2])
            gates = flow.matmul(x_t, self.W) + flow.matmul(h_t, self.U) + self.bias
            i_t, f_t, g_t, o_t = (
                flow.sigmoid(gates[:, :HS]),
                flow.sigmoid(gates[:, HS : HS * 2]),
                flow.tanh(gates[:, HS * 2 : HS * 3]),
                flow.sigmoid(gates[:, HS * 3 :]),
            )
            c_t = f_t * c_t + i_t * g_t
            h_t = o_t * flow.tanh(c_t)
            hidden_seq.append(h_t.unsqueeze(0))
        hidden_seq = flow.cat(hidden_seq, dim=0)
        return hidden_seq, (h_t, c_t)
