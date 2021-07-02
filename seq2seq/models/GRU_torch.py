import torch
import torch.nn as nn
from math import sqrt


class GRU_pytorch(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, bias=True, batch_first=True, dropout=0, bidirectional=False):
        super(GRU_pytorch, self).__init__()
        self.gru = GRU_cell_pytorch(input_size, hidden_size)

    def forward(self, input, h_0=None):
        gru_out, hidden = self.gru(input, h_0)
        return gru_out, hidden


class GRU_cell_pytorch(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(GRU_cell_pytorch, self).__init__()
        low, upper = -sqrt(1 / hidden_size), sqrt(1 / hidden_size)

        self.input_size = input_size
        self.hidden_size = hidden_size

        self.inp_W = nn.Parameter(torch.Tensor(input_size, hidden_size * 3))
        self.hid_W = nn.Parameter(torch.Tensor(hidden_size, hidden_size * 3))
        self.inp_b = nn.Parameter(torch.Tensor(hidden_size * 3))
        self.hid_b = nn.Parameter(torch.Tensor(hidden_size * 3))

        self.init_weight(low, upper)

    def init_weight(self, low, upper):
        self.inp_W.data = torch.rand(self.input_size, self.hidden_size * 3) * (upper - low) + low
        self.hid_W.data = torch.rand(self.hidden_size, self.hidden_size * 3) * (upper - low) + low
        self.inp_b.data = torch.rand(self.hidden_size * 3) * (upper - low) + low
        self.hid_b.data = torch.rand(self.hidden_size * 3) * (upper - low) + low

    def forward(self, x, hidden=None):
        # x.shape=[batch,seq_len,embed_size]  batch_first
        batch_size, seq_len, _ = x.size()
        H_S = self.hidden_size
        hidden_seq = []

        if hidden is None:
            h_t = torch.zeros(batch_size, self.hidden_size)
        else:
            h_t = hidden

        for t in range(seq_len):
            x_t = x[:, t, :]
            gates_1 = torch.matmul(x_t, self.inp_W) + self.inp_b
            gates_2 = torch.matmul(h_t, self.hid_W) + self.hid_b

            r_gate = torch.sigmoid(gates_1[:, :H_S] + gates_2[:, :H_S])
            z_gate = torch.sigmoid(gates_1[:, H_S:H_S * 2] + gates_2[:, H_S:H_S * 2])
            h_t_ = torch.tanh(gates_1[:, H_S * 2:H_S * 3] + r_gate * gates_2[:, H_S * 2:H_S * 3])
            h_t = (1 - z_gate) * h_t_ + z_gate * h_t
            hidden_seq.append(h_t.unsqueeze(1))

        hidden_seq = torch.cat(hidden_seq, dim=1)
        return hidden_seq, h_t