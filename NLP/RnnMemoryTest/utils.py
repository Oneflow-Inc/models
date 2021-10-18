import math
import numpy as np
import oneflow as flow
from oneflow import nn
import itertools

fetch_col = 'temperature'


def sliding_window(seq, window_size):
    result = []
    for i in range(len(seq) - window_size):
        result.append(seq[i:i+window_size])
    return result


def gen_set(raw_df, window_size):
    train_set, test_set = [], []
    for sensor_index, group in raw_df.groupby('sensor_index'):
        full_seq = group[fetch_col].interpolate(method='linear', limit=3, limit_area='outside')
        full_len = full_seq.shape[0]
        train_seq, test_seq = full_seq.iloc[:int(full_len * 0.8)].to_list(),\
                              full_seq.iloc[int(full_len * 0.8):].to_list()
        train_set += sliding_window(train_seq, window_size=window_size)
        test_set += sliding_window(test_seq, window_size=window_size)

    train_set, test_set = np.array(train_set), np.array(test_set)
    train_set, test_set = (item[~np.isnan(item).any(axis=1)] for item in (train_set, test_set))
    return train_set, test_set


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


def train_rnn(denorm_func, rnn_model, output_model, train_set, test_set, device, num_epoch, batch_size, lr):
    rnn_model = rnn_model.to(device)
    output_model = output_model.to(device)

    loss_func = nn.MSELoss()
    optimizer = flow.optim.Adam(itertools.chain(rnn_model.parameters(),output_model.parameters()), lr=lr)


    def pre_batch(batch):
        batch = flow.tensor(batch,dtype=flow.float32).to(device)  # (batch, seq_len)
        x, label = batch[:, :batch.size(-1)-1], batch[:, batch.size(-1)-1]

        out, hc = rnn_model(x.unsqueeze(-1))
        out = output_model(out[:, out.size(1)-1, :])
        prediction = out.squeeze(-1)  # (batch)
        return prediction, label

    loss_log = []
    score_log = []
    trained_batches = 0
    for epoch in range(num_epoch):
        for batch in next_batch(train_set, batch_size=batch_size):
            prediction, label = pre_batch(batch)
            loss = loss_func(prediction, label)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            loss_log.append(loss.detach().to('cpu').numpy().tolist())
            trained_batches += 1
            if trained_batches % 100 == 0: 
                all_prediction = []
                for batch in next_batch(test_set, batch_size=64):
                    prediction, label = pre_batch(batch)
                    all_prediction.append(prediction.detach().to('cpu').numpy())
                all_prediction = np.concatenate(all_prediction)
                index = np.shape(test_set)[1]-1
                all_label = test_set[:, index]
                all_prediction = denorm_func(all_prediction, fetch_col)
                all_label = denorm_func(all_label, fetch_col)
    return score_log, loss_log

