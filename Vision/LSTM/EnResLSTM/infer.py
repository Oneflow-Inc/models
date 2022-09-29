import oneflow as torch
import oneflow.nn as nn
import oneflow.optim as optim
import oneflow.utils.data as Data
from oneflow.nn import init
import oneflow.nn.functional as F

import numpy as np
from matplotlib import pyplot as plt
import time
import pandas as pd
from sklearn.utils import shuffle
import math
from sklearn.metrics import mean_squared_error as mse, mean_absolute_error as mae

data = np.load("dataset.npz", allow_pickle=True)
train_x = data['train_x']
train_y = data['train_y']
test_x = data['test_x']

# 数据已经提前进行了滑窗划分，而且label和自变量也是分开的。所以只需要分训练测试集就行。
device = torch.device("cuda") if torch.cuda.is_available() else torch.device(
    "cpu")
# device = torch.device("cpu")
print(device)
print(torch.cuda.is_available())
print(torch.__version__)

data_set = np.concatenate((train_x, train_y), axis=1)
data_num = data_set.shape[0]
train_set = data_set[:int(data_num*0.8),:,:]
test_set = data_set[int(data_num*0.8):,:,:]

print('train_set.shape:', train_set.shape)
print('test_set.shape:', test_set.shape)

def mape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    non_zero_index = (y_true > 0)
    y_true = y_true[non_zero_index]
    y_pred = y_pred[non_zero_index]

    mape = np.abs((y_true - y_pred) / y_true)
    mape[np.isinf(mape)] = 0
    return np.mean(mape) * 100

def next_batch(data, batch_size):
    data_length = data.shape[0]
    num_batches = math.ceil(data_length / batch_size)
    for batch_index in range(num_batches):
        start_index = batch_index * batch_size
        end_index = min((batch_index + 1) * batch_size, data_length)
        yield data[start_index:end_index]

# dropout_parameter = 0.3
# 定义LSTM输出之后的网络层，目的是要让数据升维成原数据的维度
# 改变相关的参数也可以作为数据预处理的模型
class OutputNet(torch.nn.Module):
    def __init__(self, LSTM_dim, hidden_dim1, hidden_dim2, data_dim):
        super(OutputNet, self).__init__()
        self.fc1 = torch.nn.Linear(LSTM_dim, hidden_dim1)
        self.bn1 = torch.nn.LayerNorm(hidden_dim1)
        self.fc2 = torch.nn.Linear(hidden_dim1, hidden_dim2)
        self.bn2 = torch.nn.LayerNorm(hidden_dim2)
        self.fc3 = torch.nn.Linear(hidden_dim2, hidden_dim2)
        self.bn3 = torch.nn.LayerNorm(hidden_dim2)
        self.fc4 = torch.nn.Linear(hidden_dim2, data_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        # x = self.bn1(x) # 过标准化层
        x = F.relu(self.fc2(x))
        # x = self.bn2(x)
        x = F.relu(self.fc3(x))
        # x = self.bn3(x)
        x = (self.fc4(x))
        return x

#初始化模型
print("------初始化模型Begin------")
for i in range(12):
    locals()['Convolution_Net_'+str(i)] = OutputNet(LSTM_dim=221, hidden_dim1=1024, hidden_dim2=256, data_dim=256).to(device)
    locals()['lstm_'+str(i)] = nn.LSTM(input_size=256, hidden_size=256, num_layers=3, batch_first=True).to(device)
    locals()['output_model_'+str(i)] = OutputNet(LSTM_dim=256, hidden_dim1=1024, hidden_dim2=512, data_dim=221).to(device)
print("------初始化模型End------")

print("------加载模型Begin------")
# 加载已经训练好的模型
for load_index in range(12):
    params = torch.load("./model_para/oneResEnConvolution_Net_"+str(load_index)+".pth") # 加载参数
    locals()['Convolution_Net_'+str(load_index)].load_state_dict(params) # 应用到网络结构中
    params = torch.load("./model_para/oneResEnLstm_"+str(load_index)+".pth") # 加载参数
    locals()['lstm_'+str(load_index)].load_state_dict(params) # 应用到网络结构中
    params = torch.load("./model_para/oneResEnOutput_model_"+str(load_index)+".pth") # 加载参数
    locals()['output_model_'+str(load_index)].load_state_dict(params) # 应用到网络结构中
print("------加载模型End------")

# 输出结果文件
print("------输出结果文件Begin------")
batch_size = 128
all_prediction = []
for batch in next_batch(test_x, batch_size=batch_size):
    x, final_out = batch[:, :12, :], np.zeros_like(batch[:, :12, :])
    x = torch.from_numpy(x).float().to(device)
    for model_index in range(12):
        # 使用短序列的前12个值作为历史，最后一个值作为预测值。
        x_for = x
        x_for = locals()['Convolution_Net_'+str(model_index)](x_for)
        out_for, _ = locals()['lstm_'+str(model_index)](x_for.view((x_for.shape[0], 12, -1)))  # out: (batch_size, seq_len, hidden_size)
        out_for = locals()['output_model_'+str(model_index)](out_for[:, -1, :]).squeeze(dim=1)
        final_out[:, model_index, :] = (out_for+x[:, 11, :]).detach().cpu().numpy()
        x[:, :11, :] = x[:, 1:, :]
        x[:, 11, :] = (out_for+x[:, 11, :]).detach()
    all_prediction.append(final_out)
all_prediction = np.concatenate(all_prediction)
print(all_prediction.shape)
np.savez("./data/19211278-XueruiSu-HuaiyuWan-OneFlow-Score-48.npz", test_y=all_prediction)
print("------输出结果文件End------")

# 下面是得分48的结果，可以拿这个做baseline
data_y = np.load("./data/19211278 苏学睿 万怀宇 oneflow 48.npz", allow_pickle=True)
test_y = data_y['test_y']
print("test_y.shape:", np.array(test_y).shape)
rmse_score = math.sqrt(mse(test_y.reshape(-1, 12*221), all_prediction.reshape(-1, 12*221) ))
mae_score = mae(test_y.reshape(-1, 12*221), all_prediction.reshape(-1, 12*221) )
mape_score = mape(test_y.reshape(-1, 12*221), all_prediction.reshape(-1, 12*221) )
print('test: test_rmse_loss %.6f,test_mae_loss %.6f,test_mape_loss %.6f'%(rmse_score, mae_score, mape_score))    

