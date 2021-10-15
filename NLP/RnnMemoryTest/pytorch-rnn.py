
import math

import torch
from torch import nn
import numpy as np

from dataset.dataset import *



class MyRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.hidden_size = hidden_size
        
        self.w_h = nn.Parameter(torch.rand(input_size, hidden_size))
        self.u_h = nn.Parameter(torch.rand(hidden_size, hidden_size))
        self.b_h = nn.Parameter(torch.zeros(hidden_size))
        
        self.w_y = nn.Parameter(torch.rand(hidden_size, output_size))
        self.b_y = nn.Parameter(torch.zeros(output_size))
        
        self.tanh = nn.Tanh()
        self.leaky_relu = nn.LeakyReLU()
        
        for param in self.parameters():
            if param.dim() > 1:
                nn.init.xavier_uniform_(param)
                
    def forward(self, x):
        batch_size = x.size(0)
        seq_len = x.size(1)
        
        h = torch.zeros(batch_size, self.hidden_size).to(x.device)
        y_list = []
        for i in range(seq_len):
            h = self.tanh(torch.matmul(x[:, i, :], self.w_h) + 
                             torch.matmul(h, self.u_h) + self.b_h)  # (batch_size, hidden_size)
            y = self.leaky_relu(torch.matmul(h, self.w_y) + self.b_y)  # (batch_size, output_size)
            y_list.append(y)
        return h, torch.stack(y_list, dim=1)



device = 'cuda:0'  
batch_size = 64
seq_len = 12
input_size = 2
hidden_size = 32
output_size = 1

x = torch.rand(batch_size, seq_len, input_size).to(device)
rnn = MyRNN(input_size, hidden_size, output_size).to(device)
hidden, y = rnn(x)
print(hidden.shape, y.shape)

dataset = KrakowDataset()
raw_df = dataset.data
raw_df.dropna().head()



def sliding_window(seq, window_size):
    result = []
    for i in range(len(seq) - window_size):
        result.append(seq[i:i+window_size])
    return result

fetch_col = 'temperature'
train_set, test_set = [], []
for sensor_index, group in raw_df.groupby('sensor_index'):
    full_seq = group[fetch_col].interpolate(method='linear', limit=3, limit_area='outside')
    full_len = full_seq.shape[0]
    train_seq, test_seq = full_seq.iloc[:int(full_len * 0.8)].to_list(),                          full_seq.iloc[int(full_len * 0.8):].to_list()
    train_set += sliding_window(train_seq, window_size=13)
    test_set += sliding_window(test_seq, window_size=13)

train_set, test_set = np.array(train_set), np.array(test_set)
train_set, test_set = (item[~np.isnan(item).any(axis=1)] for item in (train_set, test_set))
print(train_set.shape, test_set.shape)



device = 'cuda:0'
model = MyRNN(input_size=1, hidden_size=32, output_size=1).to(device)

loss_func = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)



from sklearn.metrics import mean_squared_error as mse, mean_absolute_error as mae


def mape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    non_zero_index = (y_true > 0)
    y_true = y_true[non_zero_index]
    y_pred = y_pred[non_zero_index]

    mape = np.abs((y_true - y_pred) / y_true)
    mape[np.isinf(mape)] = 0
    return np.mean(mape) * 100



def next_batch(data, batch_size):
    data_length = len(data)
    num_batches = math.ceil(data_length / batch_size)
    for batch_index in range(num_batches):
        start_index = batch_index * batch_size
        end_index = min((batch_index + 1) * batch_size, data_length)
        yield data[start_index:end_index]


from sklearn.utils import shuffle

loss_log = []
score_log = []
trained_batches = 0
for epoch in range(10):
    print('epoch:',epoch)
    for batch in next_batch(shuffle(train_set), batch_size=64):
        batch = torch.from_numpy(batch).float().to(device)  # (batch, seq_len)
        x, label = batch[:, :12], batch[:, -1]
        
        hidden, out = model(x.unsqueeze(-1))
        prediction = out[:, -1, :].squeeze(-1)  # (batch)
        
        loss = loss_func(prediction, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_log.append(loss.detach().cpu().numpy().tolist())
        trained_batches += 1
        
        if trained_batches % 100 == 0:
            all_prediction = []
            for batch in next_batch(test_set, batch_size=64):
                batch = torch.from_numpy(batch).float().to(device)  # (batch, seq_len)
                x, label = batch[:, :12], batch[:, -1]
                
                hidden, out = model(x.unsqueeze(-1))
                prediction = out[:, -1, :].squeeze(-1)  # (batch)
                all_prediction.append(prediction.detach().cpu().numpy())
            all_prediction = np.concatenate(all_prediction)
            all_label = test_set[:, -1]
            all_prediction = dataset.denormalize(all_prediction, fetch_col)
            all_label = dataset.denormalize(all_label, fetch_col)
            rmse_score = math.sqrt(mse(all_label, all_prediction))
            mae_score = mae(all_label, all_prediction)
            mape_score = mape(all_label, all_prediction)
            score_log.append([rmse_score, mae_score, mape_score])
            print('RMSE: %.4f, MAE: %.4f, MAPE: %.4f' % (rmse_score, mae_score, mape_score))

best_score = np.min(score_log, axis=0)



from matplotlib import pyplot as plt

plt.figure(figsize=(10, 5), dpi=300)
plt.plot(loss_log, linewidth=1)
plt.title('Loss Value')
plt.xlabel('Number of batches')
plt.show()


score_log = np.array(score_log)

plt.figure(figsize=(10, 6), dpi=300)
plt.subplot(2, 2, 1)
plt.plot(score_log[:, 0], c='#d28ad4')
plt.ylabel('RMSE')

plt.subplot(2, 2, 2)
plt.plot(score_log[:, 1], c='#e765eb')
plt.ylabel('MAE')

plt.subplot(2, 2, 3)
plt.plot(score_log[:, 2], c='#6b016d')
plt.ylabel('MAPE')

plt.show()


# In[12]:


print('Best score:', best_score)



from math import sqrt

class RNN(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        nonlinearity: str = "tanh",
        bias: bool = True,
        batch_first: bool = False,
        dropout: float = 0,
        bidirectional: bool = False,
    ):

        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.nonlinearity = nonlinearity
        self.bias = bias
        self.batch_first = batch_first
        self.dropout = dropout
        self.bidirectional = bidirectional
        num_directions = 2 if bidirectional else 1
        gate_size = hidden_size
        self.drop = nn.Dropout(self.dropout)

        if self.nonlinearity == "tanh":
            self.act = nn.Tanh()
        elif self.nonlinearity == "relu":
            self.act = nn.ReLU()
        else:
            raise ValueError("Unknown nonlinearity '{}'".format(self.nonlinearity))

        for layer in range(num_layers):
            for direction in range(num_directions):

                real_hidden_size = hidden_size
                layer_input_size = (
                    input_size if layer == 0 else real_hidden_size * num_directions
                )

                w_ih = nn.Parameter(torch.Tensor(layer_input_size, gate_size))
                w_hh = nn.Parameter(torch.Tensor(real_hidden_size, gate_size))
                b_ih = nn.Parameter(torch.Tensor(gate_size))
                b_hh = nn.Parameter(torch.Tensor(gate_size))

                layer_params = ()

                if bias:
                    layer_params = (w_ih, w_hh, b_ih, b_hh)
                else:
                    layer_params = (w_ih, w_hh)

                suffix = "_reverse" if direction == 1 else ""
                param_names = ["weight_ih_l{}{}", "weight_hh_l{}{}"]
                if bias:
                    param_names += ["bias_ih_l{}{}", "bias_hh_l{}{}"]
                param_names = [x.format(layer, suffix) for x in param_names]

                for name, param in zip(param_names, layer_params):
                    setattr(self, name, param)

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def permute_tensor(self, input):
        return input.permute(1, 0, 2)

    def forward(self, input, h_0=None):
        if self.batch_first == False:
            input = self.permute_tensor(input)

        D = 2 if self.bidirectional else 1
        num_layers = self.num_layers
        batch_size, seq_len, _ = input.size()

        if h_0 is None:
            h_t = torch.zeros(
                (D * num_layers, batch_size, self.hidden_size),
                dtype=input.dtype,
                device=input.device,
            )
        else:
            h_t = h_0

        if self.bidirectional:
            if h_0 is None:
                h_t_f = h_t[:num_layers, :, :]
                h_t_b = h_t[num_layers:, :, :]
            else:
                h_t_f = torch.cat(
                    [
                        h_t[l, :, :].unsqueeze(0)
                        for l in range(h_t.size(0))
                        if l % 2 == 0
                    ],
                    dim=0,
                )
                h_t_b = torch.cat(
                    [
                        h_t[l, :, :].unsqueeze(0)
                        for l in range(h_t.size(0))
                        if l % 2 != 0
                    ],
                    dim=0,
                )
        else:
            h_t_f = h_t

        layer_hidden = []

        for layer in range(self.num_layers):
            hidden_seq_f = []
            if self.bidirectional:
                hidden_seq_b = []

            hid_t_f = h_t_f[layer, :, :]
            if self.bidirectional:
                hid_t_b = h_t_b[layer, :, :]

            for t in range(seq_len):
                if layer == 0:
                    x_t_f = input[:, t, :]
                    if self.bidirectional:
                        x_t_b = input[:, seq_len - 1 - t, :]
                else:
                    x_t_f = hidden_seq[:, t, :]
                    if self.bidirectional:
                        x_t_b = hidden_seq[:, seq_len - 1 - t, :]

                hy1_f = torch.matmul(
                    x_t_f, getattr(self, "weight_ih_l{}{}".format(layer, "")),
                )
                hy2_f = torch.matmul(
                    hid_t_f, getattr(self, "weight_hh_l{}{}".format(layer, "")),
                )

                if self.bias:
                    hy1_f += getattr(self, "bias_ih_l{}{}".format(layer, ""))
                    hy2_f += getattr(self, "bias_hh_l{}{}".format(layer, ""))
                hid_t_f = self.act(hy1_f + hy2_f)

                hidden_seq_f.append(hid_t_f.unsqueeze(1))

                if self.bidirectional:
                    hy1_b = torch.matmul(
                        x_t_b,
                        getattr(self, "weight_ih_l{}{}".format(layer, "_reverse")),
                    )
                    hy2_b = torch.matmul(
                        hid_t_b,
                        getattr(self, "weight_hh_l{}{}".format(layer, "_reverse")),
                    )
                    if self.bias:
                        hy1_b += getattr(
                            self, "bias_ih_l{}{}".format(layer, "_reverse")
                        )
                        hy2_b += getattr(
                            self, "bias_hh_l{}{}".format(layer, "_reverse")
                        )
                    hid_t_b = self.act(hy1_b + hy2_b)

                    hidden_seq_b.insert(0, hid_t_b.unsqueeze(1))

            hidden_seq_f = torch.cat(hidden_seq_f, dim=1)
            if self.bidirectional:
                hidden_seq_b = torch.cat(hidden_seq_b, dim=1)

            if self.dropout != 0 and layer != self.num_layers - 1:
                hidden_seq_f = self.drop(hidden_seq_f)
                if self.bidirectional:
                    hidden_seq_b = self.drop(hidden_seq_b)

            if self.bidirectional:
                hidden_seq = torch.cat([hidden_seq_f, hidden_seq_b], dim=2)
            else:
                hidden_seq = hidden_seq_f

            if self.bidirectional:
                h_t = torch.cat([hid_t_f.unsqueeze(0), hid_t_b.unsqueeze(0)], dim=0)
            else:
                h_t = hid_t_f.unsqueeze(0)

            layer_hidden.append(h_t)

        h_t = torch.cat(layer_hidden, dim=0)

        if self.batch_first == False:
            hidden_seq = self.permute_tensor(hidden_seq)

        return hidden_seq, h_t


bi_rnn = RNN(input_size=2, hidden_size=64, num_layers=4, bidirectional=True, batch_first=True)


device = 'cuda:0'
batch_size = 32
seq_len = 12
input_size = 2

x = torch.randn(batch_size, seq_len, input_size).to(device)
bi_rnn = bi_rnn.to(device)
output, hidden = bi_rnn(x)
print(output.shape, hidden.shape)


device = 'cuda:0'

rnn = RNN(input_size=1, hidden_size=32, num_layers=1, batch_first=True).to(device)
out_linear = nn.Sequential(nn.Linear(32, 1), nn.LeakyReLU()).to(device)

loss_func = nn.MSELoss()
optimizer = torch.optim.Adam(list(rnn.parameters()) + list(out_linear.parameters()), lr=0.0001)

loss_log = []
score_log = []
trained_batches = 0
for epoch in range(10):
    for batch in next_batch(shuffle(train_set), batch_size=64):
        batch = torch.from_numpy(batch).float().to(device)  # (batch, seq_len)
        x, label = batch[:, :12], batch[:, -1]
        
        out, hidden = rnn(x.unsqueeze(-1))  # out: (batch_size, seq_len, hidden_size)
        out = out_linear(out[:, -1, :])
        prediction = out.squeeze(-1)  # (batch)
        
        loss = loss_func(prediction, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_log.append(loss.detach().cpu().numpy().tolist())
        trained_batches += 1
        
        if trained_batches % 100 == 0:
            all_prediction = []
            for batch in next_batch(test_set, batch_size=64):
                batch = torch.from_numpy(batch).float().to(device)  # (batch, seq_len)
                x, label = batch[:, :12], batch[:, -1]
                
                out, hidden = rnn(x.unsqueeze(-1))  # out: (batch_size, seq_len, hidden_size)
                out = out_linear(out[:, -1, :])
                prediction = out.squeeze(-1)  # (batch)
                all_prediction.append(prediction.detach().cpu().numpy())
            all_prediction = np.concatenate(all_prediction)
            all_label = test_set[:, -1]
            all_prediction = dataset.denormalize(all_prediction, fetch_col)
            all_label = dataset.denormalize(all_label, fetch_col)
            rmse_score = math.sqrt(mse(all_label, all_prediction))
            mae_score = mae(all_label, all_prediction)
            mape_score = mape(all_label, all_prediction)
            score_log.append([rmse_score, mae_score, mape_score])
            print('RMSE: %.4f, MAE: %.4f, MAPE: %.4f' % (rmse_score, mae_score, mape_score))

best_score = np.min(score_log, axis=0)


# In[17]:


plt.figure(figsize=(10, 5), dpi=300)
plt.plot(loss_log, linewidth=1)
plt.title('Loss Value')
plt.xlabel('Number of batches')
plt.show()


# In[18]:


score_log = np.array(score_log)

plt.figure(figsize=(10, 6), dpi=300)
plt.subplot(2, 2, 1)
plt.plot(score_log[:, 0], c='#d28ad4')
plt.ylabel('RMSE')

plt.subplot(2, 2, 2)
plt.plot(score_log[:, 1], c='#e765eb')
plt.ylabel('MAE')

plt.subplot(2, 2, 3)
plt.plot(score_log[:, 2], c='#6b016d')
plt.ylabel('MAPE')

plt.show()

print('Best score:', best_score)




