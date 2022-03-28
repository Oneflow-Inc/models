import pandas as pd
import oneflow as flow
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.preprocessing import LabelEncoder
from utils import SparseFeat, get_feature_names
from model import DCN
from config import *


def datareader():
    train_data = pd.read_csv("./Frappe_x1/train.csv")
    valid_data = pd.read_csv("./Frappe_x1/valid.csv")
    test_data = pd.read_csv("./Frappe_x1/test.csv")

    train_nums = len(train_data)
    valid_nums = len(valid_data)
    test_nums = len(test_data)

    all_data = (
        train_data.append(valid_data.append(test_data)).reset_index(drop=True).copy()
    )

    return all_data, train_nums, valid_nums, test_nums


def labelencoder():
    for feat in sparse_features:
        lbe = LabelEncoder()
        all_data[feat] = lbe.fit_transform(all_data[feat])


def tester(args, model_dir):
    model = DCN(
        linear_feature_columns=linear_feature_columns,
        dnn_feature_columns=dnn_feature_columns,
        cross_num=args.cross_num,
        cross_parameterization=args.cross_parameterization,
        dnn_hidden_units=[int(x) for x in args.dnn_hidden_units.split(",")],
        l2_reg_embedding=args.l2_reg_embedding,
        l2_reg_cross=args.l2_reg_cross,
        l2_reg_dnn=args.l2_reg_dnn,
        seed=args.seed,
        dnn_dropout=args.dnn_dropout,
        dnn_activation=args.dnn_activation,
        dnn_use_bn=args.dnn_use_bn,
        task=args.task,
        device=device,
        gpus=None,
    )

    print(model)
    print(model.state_dict)

    model.load_state_dict(flow.load(model_dir))
    print("load model ...")

    pred_ans = model.predict(test_model_input, 256)
    print("")
    print("test LogLoss", round(log_loss(test[target].values, pred_ans), 4))
    print("test AUC", round(roc_auc_score(test[target].values, pred_ans), 4))


if __name__ == "__main__":

    args = get_args()

    all_data, train_nums, valid_nums, test_nums = datareader()
    sparse_features = list(all_data.columns[1:])
    target = ["label"]
    # 1.Label Encoding for sparse features,and do simple Transformation for dense features
    labelencoder()

    # 2.count #unique features for each sparse field,and record dense feature field name
    fixlen_feature_columns = [
        SparseFeat(feat, all_data[feat].nunique(), embedding_dim=args.embedding_dim)
        for feat in sparse_features
    ]

    dnn_feature_columns = fixlen_feature_columns
    linear_feature_columns = fixlen_feature_columns

    feature_names = get_feature_names(linear_feature_columns + dnn_feature_columns)

    # 3.generate input data for model
    train = all_data.iloc[:train_nums, :]
    valid = all_data.iloc[train_nums : train_nums + valid_nums, :]
    test = all_data.iloc[
        train_nums + valid_nums : train_nums + valid_nums + test_nums, :
    ]

    train_model_input = {name: train[name] for name in feature_names}
    valid_model_input = {name: valid[name] for name in feature_names}
    test_model_input = {name: test[name] for name in feature_names}

    # 4.Define Model,train,predict and evaluate

    device = "cuda:0" if flow.cuda.is_available() else "cpu"

    model_dir_1 = "./log/models/frappe"
    model_dir_2 = "./models-torch2flow/frappe"

    model = DCN(
        linear_feature_columns=linear_feature_columns,
        dnn_feature_columns=dnn_feature_columns,
        cross_num=args.cross_num,
        cross_parameterization=args.cross_parameterization,
        dnn_hidden_units=[int(x) for x in args.dnn_hidden_units.split(",")],
        l2_reg_embedding=args.l2_reg_embedding,
        l2_reg_cross=args.l2_reg_cross,
        l2_reg_dnn=args.l2_reg_dnn,
        seed=args.seed,
        dnn_dropout=args.dnn_dropout,
        dnn_activation=args.dnn_activation,
        dnn_use_bn=args.dnn_use_bn,
        task=args.task,
        device=device,
        gpus=None,
    )

    print(model.state_dict)

    model.load_state_dict(flow.load(model_dir_2))

    print(model.state_dict)
    print("load model ...")

    pred_ans = model.predict(test_model_input, 256)
    print("")
    print("test LogLoss", round(log_loss(test[target].values, pred_ans), 4))
    print("test AUC", round(roc_auc_score(test[target].values, pred_ans), 4))
