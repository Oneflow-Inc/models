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

#读取数据集，进行划分
def sliding_window(seq,window_size):
    result = []
    for i in range(seq.shape[0]- window_size):
        result.append(seq[i: i+window_size])
    result = np.array(result)
    return result

def MLPreshape(set):
    set = set.reshape((set.shape[0],set.shape[1],-1))
    return set

def MLPreshape_inverse(set):
    set = set.reshape((set.shape[0],set.shape[1],16,16))
    return set

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

# 加载已经训练好的模型
for load_index in range(12):
    params = torch.load("./model_para/oneResEnConvolution_Net_"+str(load_index)+".pth") # 加载参数
    locals()['Convolution_Net_'+str(load_index)].load_state_dict(params) # 应用到网络结构中
    params = torch.load("./model_para/oneResEnLstm_"+str(load_index)+".pth") # 加载参数
    locals()['lstm_'+str(load_index)].load_state_dict(params) # 应用到网络结构中
    params = torch.load("./model_para/oneResEnOutput_model_"+str(load_index)+".pth") # 加载参数
    locals()['output_model_'+str(load_index)].load_state_dict(params) # 应用到网络结构中

loss_func = nn.MSELoss()
parameters = list(locals()['Convolution_Net_'+str(11)].parameters()) + list(locals()['lstm_'+str(11)].parameters()) + list(locals()['output_model_'+str(11)].parameters())
for index in range(11):
    parameters = parameters + list(locals()['Convolution_Net_'+str(index)].parameters()) + list(locals()['lstm_'+str(index)].parameters()) + list(locals()['output_model_'+str(index)].parameters())
optimizer = torch.optim.Adam(parameters, lr=0.0001, weight_decay=1e-4)

# 训练模块
# 这个部分如果是代码审核可以直接把它注释掉运行下面“输出结果文件”的部分
# 保存打印文件
f = open("./data/FlowVersion02res-oneflow.txt", 'a+')
train_log = []
test_log = []
#开始时间
timestart = time.time()
trained_batches = 0 #记录多少个batch
batch_size = 128
print("------模型训练Begin------")
for epoch in range(160):
    if epoch >= 80 and epoch < 100 :
        batch_size = 128
        optimizer = torch.optim.Adam(parameters, lr=1e-4/2, weight_decay=1e-4/2)
    elif epoch >= 100 and epoch < 120 :
        batch_size = 64
        optimizer = torch.optim.Adam(parameters, lr=1e-4/10, weight_decay=1e-4/10)
    elif epoch >= 120 and epoch < 160:
        batch_size = 32
        optimizer = torch.optim.Adam(parameters, lr=1e-4/40, weight_decay=1e-4/40)    
    total_1oss = 0 #记录Loss
    for batch in next_batch(shuffle(data_set), batch_size=batch_size):
        #每一个batch的开始时间
        batchstart = time.time()
        x, label_all= batch[:, :12, :], batch[:, 12:, :]
        loss_sum = torch.tensor(0.).float().to(device)
        x = torch.from_numpy(x).float().to(device)
        for model_index in range(12):
            # 使用短序列的前12个值作为历史，最后一个值作为预测值。
            x_for = x
            label = torch.from_numpy(label_all[:, model_index, :]).float().to(device)
            x_for = locals()['Convolution_Net_'+str(model_index)](x_for)
            out_for, _ = locals()['lstm_'+str(model_index)](x_for.view((x_for.shape[0], 12, -1)))  
            # out_for: (batch_size, seq_len, hidden_size)
            out_for = locals()['output_model_'+str(model_index)](out_for[:, -1, :]).squeeze(dim=1)
            loss = loss_func(out_for, label - x[:, 11, :])
            loss_sum += loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            x[:, :11, :] = x[:, 1:, :]
            x[:, 11, :] = (out_for+x[:, 11, :]).detach()

        trained_batches += 1
        train_log.append(loss_sum.detach().cpu().numpy().tolist())
        train_batch_time = (time.time() - batchstart)

        # 每训练一定数量的batch，就在测试集上测试模型效果。
        if trained_batches % 300 == 0:
            print('epoch %d, batch %d, train_loss %.6f,Time used %.6fs'%(epoch, trained_batches, loss,train_batch_time))
            print('epoch %d, batch %d, train_loss %.6f,Time used %.6fs'%(epoch, trained_batches, loss,train_batch_time), file=f)
            #每一个batch的开始时间
            batch_test_start = time.time()
            #在每个epoch上测试
            all_prediction = []
            
            for batch in next_batch(test_set, batch_size=batch_size):
                x, final_out = batch[:, :12, :], np.zeros_like(batch[:, :12, :])
                x = torch.from_numpy(x).float().to(device)
                for model_index in range(12):
                    # 使用短序列的前12个值作为历史，最后一个值作为预测值。
                    x_for = x
                    x_for = locals()['Convolution_Net_'+str(model_index)](x_for)
                    out_for, _ = locals()['lstm_'+str(model_index)](x_for.view((x_for.shape[0], 12, -1)))  
                    out_for = locals()['output_model_'+str(model_index)](out_for[:, -1, :]).squeeze(dim=1)
                    final_out[:, model_index, :] = (out_for+x[:, 11, :]).detach().cpu().numpy()
                    x[:, :11, :] = x[:, 1:, :]
                    x[:, 11, :] = (out_for+x[:, 11, :]).detach()
                all_prediction.append(final_out)
            all_prediction = np.concatenate(all_prediction)
            all_label = test_set[:, 12:, :]
            rmse_score = math.sqrt(mse(all_label.reshape(-1, 12*221), all_prediction.reshape(-1, 12*221) ))
            mae_score = mae(all_label.reshape(-1, 12*221), all_prediction.reshape(-1, 12*221) )
            mape_score = mape(all_label.reshape(-1, 12*221), all_prediction.reshape(-1, 12*221) )
            
            # 计算测试指标。
            test_log.append([rmse_score, mae_score, mape_score])
            test_batch_time = (time.time() - batch_test_start)
            print('***************************test_batch %d, test_rmse_loss %.6f,test_mae_loss %.6f,test_mape_loss %.6f,Time used %.6fs'%(trained_batches, rmse_score,mae_score,mape_score,test_batch_time))
            print('***************************test_batch %d, test_rmse_loss %.6f,test_mae_loss %.6f,test_mape_loss %.6f,Time used %.6fs'%(trained_batches, rmse_score,mae_score,mape_score,test_batch_time),file=f)

#计算总时间
timesum = (time.time() - timestart)
print('The total time is %fs'%(timesum))
print('The total time is %fs'%(timesum),file=f)
f.close()
print("------模型训练End------")

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

# 保存模型
print("------保存模型参数Begin------")
for save_index in range(12):
    torch.save(locals()['Convolution_Net_'+str(save_index)].state_dict(), "./model_para/0oneResEnConvolution_Net_"+str(save_index)+".pth") # 保存参数
    torch.save(locals()['lstm_'+str(save_index)].state_dict(), "./model_para/0oneResEnLstm_"+str(save_index)+".pth") # 保存参数
    torch.save(locals()['output_model_'+str(save_index)].state_dict(), "./model_para/0oneResEnOutput_model_"+str(save_index)+".pth") # 保存参数

# 加载已经训练好的模型
for load_index in range(12):
    params = torch.load("./model_para/0oneResEnConvolution_Net_"+str(load_index)+".pth") # 加载参数
    locals()['Convolution_Net_'+str(load_index)].load_state_dict(params) # 应用到网络结构中
    params = torch.load("./model_para/0oneResEnLstm_"+str(load_index)+".pth") # 加载参数
    locals()['lstm_'+str(load_index)].load_state_dict(params) # 应用到网络结构中
    params = torch.load("./model_para/0oneResEnOutput_model_"+str(load_index)+".pth") # 加载参数
    locals()['output_model_'+str(load_index)].load_state_dict(params) # 应用到网络结构中
print("------保存模型参数End------")

print("------绘制相关指标曲线Begin------")
# 绘制训练损失函数图像：
plt.figure()
plt.plot(np.array(train_log), label='Train Loss', color='b')
plt.legend()
plt.xlabel("Batches")
plt.ylabel("Loss")
plt.title("Train Loss Figure")
plt.show()
# 训练测试指标图像
#train_loss曲线
train_x = np.linspace(0,len(train_log),len(train_log))
plt.subplot(2, 2, 1)
plt.plot(train_x,train_log,label="train_loss",linewidth=1.5)
plt.xlabel("number of batches")
plt.ylabel("loss")
plt.legend()

#test_loss曲线
x_test= np.linspace(0,len(test_log),len(test_log))
test_log = np.array(test_log)
plt.subplot(2, 2, 2)
plt.plot(x_test,test_log[:,0],label="test_rmse_loss",linewidth=1.5)
plt.xlabel("number of batches*100")
plt.ylabel("loss")
plt.legend()

#test_loss曲线
x_test= np.linspace(0,len(test_log),len(test_log))
test_log = np.array(test_log)
plt.subplot(2, 2, 3)
plt.plot(x_test,test_log[:,1],label="test_mae_loss",linewidth=1.5)
plt.xlabel("number of batches*100")
plt.ylabel("loss")
plt.legend()

#test_loss曲线
x_test= np.linspace(0,len(test_log),len(test_log))
test_log = np.array(test_log)
plt.subplot(2, 2, 4)
plt.plot(x_test,test_log[:,2],label="test_mape_loss",linewidth=1.5)
plt.xlabel("number of batches*100")
plt.ylabel("loss")
plt.legend()
plt.savefig('./figure/restrain-ensemble.jpg')
plt.show()
print("------绘制相关指标曲线End------")

