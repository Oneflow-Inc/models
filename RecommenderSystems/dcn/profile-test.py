import pandas as pd
import torch
import oneflow as flow
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.preprocessing import LabelEncoder
from utils import SparseFeat, get_feature_names
from model import DCN
from config import *
import copy

import numpy as np
import oneflow.utils.data as Data
from oneflow.utils.data import DataLoader
import time

def datareader():
    train_data = pd.read_csv('./Frappe_x1/train.csv')
    valid_data = pd.read_csv('./Frappe_x1/valid.csv')
    test_data = pd.read_csv('./Frappe_x1/test.csv')

    train_nums = len(train_data)
    valid_nums = len(valid_data)
    test_nums = len(test_data)

    all_data = train_data.append(valid_data.append(test_data)).reset_index(drop=True).copy()
    
    return all_data, train_nums, valid_nums, test_nums


def labelencoder():
    for feat in sparse_features:
        lbe = LabelEncoder()
        all_data[feat] = lbe.fit_transform(all_data[feat])

def trainer(args):
    model = DCN(linear_feature_columns=linear_feature_columns, 
                dnn_feature_columns=dnn_feature_columns,
                cross_num=args.cross_num, 
                cross_parameterization=args.cross_parameterization,               
                dnn_hidden_units=[int(x) for x in args.dnn_hidden_units.split(',')],  
                l2_reg_embedding=args.l2_reg_embedding, 
                l2_reg_cross=args.l2_reg_cross,
                l2_reg_dnn=args.l2_reg_dnn, 
                seed=args.seed, 
                dnn_dropout=args.dnn_dropout, 
                dnn_activation=args.dnn_activation, 
                dnn_use_bn=args.dnn_use_bn,                 
                task=args.task, 
                device=device, 
                gpus=None)

    model.compile(optimizer=args.optimizer, 
                loss=args.loss,
                metrics=args.metrics.split(','), 
                lr=args.lr)

    model.fit(train_model_input,
            train[target].values,
            batch_size=args.batch_size,
            epochs=args.epochs,
            verbose=args.verbose,
            validation_data=(valid_model_input,valid[target].values),
            shuffle=args.shuffle,
            model_dir=args.model_dir)


def tester(args):
    model = DCN(linear_feature_columns=linear_feature_columns, 
                dnn_feature_columns=dnn_feature_columns,
                cross_num=args.cross_num, 
                cross_parameterization=args.cross_parameterization,                 
                dnn_hidden_units=[int(x) for x in args.dnn_hidden_units.split(',')],  
                l2_reg_embedding=args.l2_reg_embedding, 
                l2_reg_cross=args.l2_reg_cross,
                l2_reg_dnn=args.l2_reg_dnn, 
                seed=args.seed, 
                dnn_dropout=args.dnn_dropout, 
                dnn_activation=args.dnn_activation, 
                dnn_use_bn=args.dnn_use_bn,                 
                task=args.task, 
                device=device, 
                gpus=None)

    model.load_state_dict(flow.load(args.model_dir))
    print('load model ...')

    pred_ans = model.predict(test_model_input, 256)
    print("")
    print("test LogLoss", round(log_loss(test[target].values, pred_ans), 4))
    print("test AUC", round(roc_auc_score(test[target].values, pred_ans), 4))


def test_with_time(model, args):
    model = model.train()
    loss_func = model.loss_func
    optim = model.optim

    x = train_model_input
    y = train[target].values

    if isinstance(x, dict):
        x = [x[feature] for feature in model.feature_index]

    for i in range(len(x)):
        if len(x[i].shape) == 1:
            x[i] = np.expand_dims(x[i], axis=1)

    train_tensor_data = Data.TensorDataset(
        flow.from_numpy(
            np.concatenate(x, axis=-1)),
        flow.from_numpy(y))

    train_loader = DataLoader(
    dataset=train_tensor_data, shuffle=args.shuffle, batch_size=args.batch_size)
    
    time_dict ={
                        'load data':0,
                        'forward':0,
                        'loss':0,
                        'zero grad':0,
                        'backward':0,
                        'optimizer step':0,
                        'loss.item':0

        }

    loss_epoch = 0
    total_loss_epoch = 0

    start_1 =time.time()
    for idx in range(20):
        x_train, y_train = iter(train_loader).__next__()

        end_1 = time.time()
        time_dict['load data']+=(end_1-start_1)
        x = x_train.to(device).float()
        y = y_train.to(device).float()

 
        start_2 = time.time()
        y_pred = model(x).squeeze()
        end_2 = time.time()
        time_dict['forward']+=(end_2-start_2)


        
        start_3 = time.time()
        optim.zero_grad()
        end_3 = time.time()
        time_dict['zero grad']+=(end_3-start_3)

        start_4 = time.time()
        loss = loss_func(y_pred, y.squeeze())

        reg_loss = model.get_regularization_loss()

        total_loss = loss + reg_loss + model.aux_loss
        end_4 = time.time()
        time_dict['loss']+=(end_4-start_4)

        start_5 = time.time()
        loss_epoch += loss.item()
        total_loss_epoch += total_loss.item()
        end_5 = time.time()
        time_dict['loss.item']+=(end_5-start_5)

        start_6 = time.time()
        total_loss.backward()
        end_6 = time.time()
        time_dict['backward']+=(end_6-start_6)


        start_7 = time.time()
        optim.step()
        end_7 = time.time()
        time_dict['optimizer step']+=(end_7-start_7)  

        start_1 = time.time()


    for time_key in time_dict:
        print('=====',time_key,'=====',time_dict[time_key]*(5/2))


def test_with_profile(model, args):
    model = model.train()
    loss_func = model.loss_func
    optim = model.optim

    x = train_model_input
    y = train[target].values

    if isinstance(x, dict):
        x = [x[feature] for feature in model.feature_index]

    for i in range(len(x)):
        if len(x[i].shape) == 1:
            x[i] = np.expand_dims(x[i], axis=1)

    train_tensor_data = Data.TensorDataset(
        flow.from_numpy(
            np.concatenate(x, axis=-1)),
        flow.from_numpy(y))

    train_loader = DataLoader(
    dataset=train_tensor_data, shuffle=args.shuffle, batch_size=args.batch_size)
    

    loss_epoch = 0
    total_loss_epoch = 0

    flow._oneflow_internal.profiler.RangePush('train DCN begin')
    for idx in range(20):

        flow._oneflow_internal.profiler.RangePush('load data')
        x_train, y_train = iter(train_loader).__next__()
        x = x_train.to(device).float()
        y = y_train.to(device).float()
        flow._oneflow_internal.profiler.RangePop()

 
        flow._oneflow_internal.profiler.RangePush('forward')
        y_pred = model(x).squeeze()
        flow._oneflow_internal.profiler.RangePop()

    
        flow._oneflow_internal.profiler.RangePush('zero grad')
        optim.zero_grad()
        flow._oneflow_internal.profiler.RangePop()


        flow._oneflow_internal.profiler.RangePush('loss')
        loss = loss_func(y_pred, y.squeeze())
        reg_loss = model.get_regularization_loss()
        total_loss = loss + reg_loss + model.aux_loss
        flow._oneflow_internal.profiler.RangePop()


        flow._oneflow_internal.profiler.RangePush('loss.item')
        loss_epoch += loss.item()
        total_loss_epoch += total_loss.item()
        flow._oneflow_internal.profiler.RangePop()


        flow._oneflow_internal.profiler.RangePush('backward')
        total_loss.backward()
        flow._oneflow_internal.profiler.RangePop()


        optim.step()
        flow._oneflow_internal.profiler.RangePop()

    flow._oneflow_internal.profiler.RangePop()


if __name__ == "__main__":

    args = get_args()

    all_data, train_nums, valid_nums, test_nums = datareader()
    sparse_features = list(all_data.columns[1:])
    target = ['label']
# 1.Label Encoding for sparse features,and do simple Transformation for dense features
    labelencoder()

# 2.count #unique features for each sparse field,and record dense feature field name
    fixlen_feature_columns = [SparseFeat(feat, all_data[feat].nunique(), embedding_dim = args.embedding_dim)
                                for feat in sparse_features]

    dnn_feature_columns = fixlen_feature_columns
    linear_feature_columns = fixlen_feature_columns

    feature_names = get_feature_names(
        linear_feature_columns + dnn_feature_columns)

# 3.generate input data for model
    train = all_data.iloc[:train_nums,:]
    valid = all_data.iloc[train_nums:train_nums+valid_nums,:]
    test = all_data.iloc[train_nums+valid_nums:train_nums+valid_nums+test_nums,:]

    train_model_input = {name: train[name] for name in feature_names}
    valid_model_input = {name: valid[name] for name in feature_names}
    test_model_input = {name: test[name] for name in feature_names}


# 4.Define Model,train,predict and evaluate

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    


    model = DCN(linear_feature_columns=linear_feature_columns, 
                    dnn_feature_columns=dnn_feature_columns,
                    cross_num=args.cross_num, 
                    cross_parameterization=args.cross_parameterization,               
                    dnn_hidden_units=[int(x) for x in args.dnn_hidden_units.split(',')],  
                    l2_reg_embedding=args.l2_reg_embedding, 
                    l2_reg_cross=args.l2_reg_cross,
                    l2_reg_dnn=args.l2_reg_dnn, 
                    seed=args.seed, 
                    dnn_dropout=args.dnn_dropout, 
                    dnn_activation=args.dnn_activation, 
                    dnn_use_bn=args.dnn_use_bn,                 
                    task=args.task, 
                    device=device, 
                    gpus=None)


    model.compile(optimizer=args.optimizer, 
                loss=args.loss,
                metrics=args.metrics.split(','), 
                lr=args.lr)


    test_with_time(model, args)

    test_with_profile(model, args)






                                                                                
