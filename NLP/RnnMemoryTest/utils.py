import math
import numpy as np
from sklearn.metrics import mean_squared_error as mse, mean_absolute_error as mae
from sklearn.utils import shuffle
import oneflow as flow
from oneflow import nn
from matplotlib import pyplot as plt
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
        for batch in next_batch(shuffle(train_set), batch_size=batch_size):
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
                rmse_score = math.sqrt(mse(all_label, all_prediction))
                mae_score = mae(all_label, all_prediction)
                mape_score = mape(all_label, all_prediction)
                score_log.append([rmse_score, mae_score, mape_score])
                print('RMSE: %.4f, MAE: %.4f, MAPE: %.4f' % (rmse_score, mae_score, mape_score))
    return score_log, loss_log


def plot_loss(loss_log):
    plt.figure(figsize=(10, 5), dpi=300)
    plt.plot(loss_log, linewidth=1)
    plt.title('Loss Value')
    plt.xlabel('Number of batches')
    plt.show()


def plot_metric(score_log):
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