import math

import oneflow as flow
import oneflow.nn as nn


class LSTMText(nn.Module):
    def __init__(self, emb_sz, emb_dim, hidden_size, nfc, n_classes, batch_size):
        super(LSTMText, self).__init__()
        self.emb_sz = emb_sz
        self.emb_dim = emb_dim
        self.n_classes = n_classes
        self.hidden_size = hidden_size
        self.nfc = nfc
        self.batch_size = batch_size
        self.bilstm = BiLSTM(emb_dim, hidden_size, batch_size=batch_size)
        self.embedding = nn.Embedding(self.emb_sz, self.emb_dim)
        self.linear = nn.Linear(hidden_size * 2 * nfc, n_classes)
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, inputs, is_train=1):
        data = self.embedding(inputs)
        data = self.bilstm(data)
        datalist = []
        for t in range(self.batch_size):
            data_t = data[:, t, :].reshape((1, -1))
            datalist.append(data_t.unsqueeze(0))
        data = flow.cat(datalist, dim=0)
        logits = self.linear(data)
        logits = logits.squeeze(1)
        logits = self.softmax(logits)
        return logits


def reverse(inputs, dim=0):
    temp = inputs.numpy()
    if dim == 0:
        temp = temp[::-1, :, :]
    elif dim == 1:
        temp = temp[:, ::-1, :]
    elif dim == 2:
        temp = temp[:, :, ::-1]
    reserve_inputs = flow.Tensor(temp).to('cuda')
    return reserve_inputs


class CustomLSTM(nn.Module):
    def __init__(self, input_sz, hidden_sz, batch_size=1, num_layers=1):
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
        """Assumes x is of shape (batch, sequence, feature)"""
        bs, seq_sz, _ = x.size()
        hidden_seq = []
        if init_states is None:
            h_t, c_t = (
                flow.zeros((bs, self.hidden_size)).to('cuda'),
                flow.zeros((bs, self.hidden_size)).to('cuda'),
            )
        else:
            h_t, c_t = init_states
        HS = self.hidden_size
        for t in range(seq_sz):
            x_t = x[:, t, :].reshape((x.shape[0], x.shape[2]))
            gates = flow.matmul(x_t, self.W) + flow.matmul(h_t, self.U) + self.bias
            i_t, f_t, g_t, o_t = (
                flow.sigmoid(gates[:, :HS]),
                flow.sigmoid(gates[:, HS: HS * 2]),
                flow.tanh(gates[:, HS * 2: HS * 3]),
                flow.sigmoid(gates[:, HS * 3:]),
            )
            c_t = f_t * c_t + i_t * g_t
            h_t = o_t * flow.tanh(c_t)
            hidden_seq.append(h_t.unsqueeze(0))
        hidden_seq = flow.cat(hidden_seq, dim=0)
        return hidden_seq, (h_t, c_t)


class BiLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, batch_size=32, num_layers=1, bi_flag=1):
        super(BiLSTM, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.num_layers = num_layers
        if bi_flag:
            self.bi_num = 2
        else:
            self.bi_num = 1
        self.biFlag = bi_flag
        self.layer1 = nn.ModuleList()
        self.layer1.append(CustomLSTM(input_dim, hidden_dim, num_layers=num_layers, batch_size=batch_size))
        if bi_flag:
            # add reverse layer.
            self.layer1.append(CustomLSTM(input_dim, hidden_dim, num_layers=num_layers, batch_size=batch_size))
    
    def init_hidden(self, batch_size):
        return (flow.zeros((batch_size, self.hidden_dim)).to('cuda'),
                flow.zeros((batch_size, self.hidden_dim)).to('cuda'))
    
    def forward(self, data):  # data: B*L*F  B = batch_size,L为seq定长，F为feature
        batch_size = data.shape[0]
        max_length = data.shape[1]
        hidden = [self.init_hidden(batch_size) for _ in range(self.bi_num)]
        reverse_inputs = reverse(data, dim=1)
        out = [data, reverse_inputs]
        for l in range(self.bi_num):
            out[l], hidden[l] = self.layer1[l](out[l], hidden[l])
            # reverse output
            if l == 1:
                out[l] = reverse(out[l], dim=0)
        if self.bi_num == 1:
            out = out[0]
        else:
            out = flow.cat(out, 2)
        return out
