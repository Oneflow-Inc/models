import argparse
from collections import OrderedDict
import pandas as pd
import torch
# import oneflow as flow
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from utils import SparseFeat, DenseFeat, get_feature_names
from model import DCN

def datareader():
    train_data = pd.read_csv('../Frappe_x1/train.csv')
    valid_data = pd.read_csv('../Frappe_x1/valid.csv')
    test_data = pd.read_csv('../Frappe_x1/test.csv')

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

    # print(model)

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
            model_path=args.model_path)


def tester(args,model_path):
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

    model.load_state_dict(torch.load(model_path))
    print('load model ...')

    pred_ans = model.predict(test_model_input, 256)
    print("")
    print("test LogLoss", round(log_loss(test[target].values, pred_ans), 4))
    print("test AUC", round(roc_auc_score(test[target].values, pred_ans), 4))

def torch2flow(model_path, model_dir):

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
    
    model.load_state_dict(torch.load(model_path))
    torch_dict = model.state_dict()
    print(torch_dict.keys())
    flow_dict = OrderedDict([])

    for key in torch_dict:
        print(key)
        if "num_batches_tracked" in key:
            print('***************remove************:',key)
            continue
        flow_dict[key] = flow.tensor(torch_dict[key].cpu())
    
    print(flow_dict.keys())

    flow.save(flow_dict, model_dir)



def get_args(print_args=True):
    parser = argparse.ArgumentParser()
    parser.add_argument("--dnn_use_bn", type=bool, default=True)
    parser.add_argument("--batch_size", type=int, default=4096)
    parser.add_argument("--cross_num", type=int, default=3)
    parser.add_argument("--cross_parameterization", type=str, default="vector")
    parser.add_argument("--dnn_activation", type=str, default="relu")
    parser.add_argument("--dnn_hidden_units", type=str, default="400,400,400")
    parser.add_argument("--embedding_dim", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--loss", type=str, default="binary_crossentropy")
    parser.add_argument("--metrics", type=str, default="binary_crossentropy,auc")
    parser.add_argument("--dnn_dropout", type=float, default=0.2)
    parser.add_argument("--l2_reg_embedding", type=int, default=0.005)    
    parser.add_argument("--l2_reg_cross", type=float, default=0.00001)
    parser.add_argument("--l2_reg_dnn", type=float, default=0)
    parser.add_argument("--optimizer", type=str, default="adam")
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--shuffle", type=bool, default=True)
    parser.add_argument("--task", type=str, default="binary")
    parser.add_argument("--model_path", type=str, default='./log/200-frappe.pth')
    parser.add_argument("--model_dir", type=str, default='../models-torch2flow/frappe')
    parser.add_argument("--verbose", type=int, default=2)
    parser.add_argument("--seed", type=int, default=2022)
    args = parser.parse_args()

    if print_args:
        _print_args(args)
    return args

def _print_args(args):
    """Print arguments."""
    print("------------------------ arguments ------------------------", flush=True)
    str_list = []
    for arg in vars(args):
        dots = "." * (48 - len(arg))
        str_list.append("  {} {} {}".format(arg, dots, getattr(args, arg)))
    for arg in sorted(str_list, key=lambda x: x.lower()):
        print(arg, flush=True)
    print("-------------------- end of arguments ---------------------", flush=True)



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
    ###

    ### init state_dict
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
            epochs=1,
            verbose=args.verbose,
            validation_data=(valid_model_input,valid[target].values),
            shuffle=args.shuffle,
            model_path=None)

    






    ###
    # trainer(args)
    # tester(args, args.model_path)
    # torch2flow(args.model_path, args.model_dir)




    ###
    # model_path = './log/200-frappe.pth'
    # model_dir = '../models-torch2flow/frappe'
    # tester(args, model_path)
    # torch2flow(model_path, model_dir)



