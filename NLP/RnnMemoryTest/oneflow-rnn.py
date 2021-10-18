import math

import oneflow as flow
from oneflow import nn

import numpy as np
from numpy.random import rand
from numpy import zeros
import itertools
from dataset.dataset import *

device = 'cuda:0'

class MyRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.w_h = nn.Parameter(flow.Tensor(rand(input_size,hidden_size)))
        self.u_h = nn.Parameter(flow.Tensor(rand(hidden_size, hidden_size)))
        self.b_h = nn.Parameter(flow.Tensor(zeros((hidden_size))))
        
        self.w_y = nn.Parameter(flow.Tensor(rand(hidden_size, output_size)))
        self.b_y = nn.Parameter(flow.Tensor(zeros(output_size)))
        
        self.tanh = nn.Tanh()
        self.leaky_relu = nn.LeakyReLU()
        
        for param in self.parameters():
            if param.dim() > 1:
                nn.init.xavier_uniform_(param)
                
    def forward(self, x):
        batch_size = x.size(0)
        seq_len = x.size(1)
        h = flow.zeros((batch_size, self.hidden_size)).to(device)
        y_list = []
        for i in range(seq_len):
            h = self.tanh(flow.matmul(x[:, i, :], self.w_h) + 
                             flow.matmul(h, self.u_h) + self.b_h)  # (batch_size, hidden_size)
            y = self.leaky_relu(flow.matmul(h, self.w_y) + self.b_y)  # (batch_size, output_size)
            y_list.append(y)
        return h, flow.stack(y_list, dim=1)

device = 'cuda:0'  
batch_size = 64
seq_len = 12
input_size = 2
hidden_size = 32
output_size = 1

x = flow.tensor(rand(batch_size,seq_len,input_size),dtype=flow.float32).to(device)
rnn = MyRNN(input_size, hidden_size, output_size).to(device)
hidden, y = rnn(x)
print(hidden.shape, y.shape)

dataset = KrakowDataset()
raw_df = dataset.data
raw_df.dropna().head()


def sliding_window(seq, window_size):
    result = []
    for i in range(len(seq) - window_size + 1):
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
optimizer = flow.optim.Adam(model.parameters(), lr=0.0001)


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


loss_log = []
score_log = []
trained_batches = 0
for epoch in range(10):
    print('epoch:',epoch)
    for batch in next_batch(train_set, batch_size=64):
        x, label = flow.Tensor(batch[:, :12]).to(device), flow.Tensor(batch[:, -1]).to(device)
        hidden, out = model(x.unsqueeze(-1))
        prediction = out[:, out.size(1)-1, :].squeeze(-1)  # (batch)
        
        loss = loss_func(prediction, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_log.append(loss.detach().to('cpu').numpy().tolist())
        trained_batches += 1
        
        if trained_batches % 100 == 0:
            all_prediction = []
            for batch in next_batch(test_set, batch_size=64):
                x, label = flow.Tensor(batch[:, :12]).to(device), flow.Tensor(batch[:, -1]).to(device)
                hidden, out = model(x.unsqueeze(-1))
                prediction = out[:, out.size(1)-1, :].squeeze(-1)  # (batch)
                all_prediction.append(prediction.detach().to('cpu').numpy())
            all_prediction = np.concatenate(all_prediction)
            all_label = test_set[:, -1]
            all_prediction = dataset.denormalize(all_prediction, fetch_col)
            all_label = dataset.denormalize(all_label, fetch_col)
            mape_score = mape(all_label, all_prediction)

best_score = np.min(score_log, axis=0)

score_log = np.array(score_log)

print('Best score:', best_score)


bi_rnn = nn.RNN(input_size=2, hidden_size=64, num_layers=4, bidirectional=True, batch_first=True)


device = 'cuda:0'
batch_size = 32
seq_len = 12
input_size = 2

x = np.random.randn(batch_size, seq_len, input_size)
x = flow.tensor(x, dtype=flow.float32).to(device)

bi_rnn = bi_rnn.to(device)
output, hidden = bi_rnn(x)
print(output.shape, hidden.shape)


device = 'cuda:0'

rnn = nn.RNN(input_size=1, hidden_size=32, num_layers=1, batch_first=True).to(device)
out_linear = nn.Sequential(nn.Linear(32, 1), nn.LeakyReLU()).to(device)

loss_func = nn.MSELoss()
optimizer = flow.optim.Adam(itertools.chain(rnn.parameters(), out_linear.parameters()), lr=0.0001)

loss_log = []
score_log = []
trained_batches = 0
for epoch in range(10):
    for batch in next_batch(train_set, batch_size=64):
        batch = flow.tensor(batch, dtype=flow.float32).to(device)   # (batch, seq_len)
        x, label = batch[:, :12], batch[:, -1]
        
        out, hidden = rnn(x.unsqueeze(-1))  # out: (batch_size, seq_len, hidden_size)
        out = out_linear(out[:, -1, :])
        prediction = out.squeeze(-1)  # (batch)
        
        loss = loss_func(prediction, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_log.append(loss.detach().to('cpu').numpy().tolist())

        trained_batches += 1
        
        if trained_batches % 100 == 0:
            all_prediction = []
            for batch in next_batch(test_set, batch_size=64):
                batch = flow.tensor(batch, dtype=flow.float32).to(device)   # (batch, seq_len)
                x, label = batch[:, :12], batch[:, -1]
                
                out, hidden = rnn(x.unsqueeze(-1))  # out: (batch_size, seq_len, hidden_size)
                out = out_linear(out[:, -1, :])
                prediction = out.squeeze(-1)  # (batch)
                all_prediction.append(prediction.detach().to('cpu').numpy())

            all_prediction = np.concatenate(all_prediction)
            all_label = test_set[:, -1]
            all_prediction = dataset.denormalize(all_prediction, fetch_col)
            all_label = dataset.denormalize(all_label, fetch_col)
            mape_score = mape(all_label, all_prediction)

