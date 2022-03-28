import pandas as pd
import oneflow as flow
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from utils import SparseFeat, DenseFeat, get_feature_names
from model import DCN


def labelencoder():
    for feat in sparse_features:
        lbe = LabelEncoder()
        data[feat] = lbe.fit_transform(data[feat])
    mms = MinMaxScaler(feature_range=(0, 1))
    data[dense_features] = mms.fit_transform(data[dense_features])


def featurescount():
    sparse_feats = [SparseFeat(feat, data[feat].nunique()) for feat in sparse_features]
    dense_feats = [DenseFeat(feat, 1,) for feat in dense_features]
    fixlen_feature_columns = sparse_feats + dense_feats
    return fixlen_feature_columns


if __name__ == "__main__":

    data = pd.read_csv("./criteo_sample.txt")

    sparse_features = ["C" + str(i) for i in range(1, 27)]
    dense_features = ["I" + str(i) for i in range(1, 14)]

    data[sparse_features] = data[sparse_features].fillna("-1",)
    data[dense_features] = data[dense_features].fillna(0,)
    target = ["label"]

    # 1.Label Encoding for sparse features,and do simple Transformation for dense features
    labelencoder()

    # 2.count #unique features for each sparse field,and record dense feature field name
    fixlen_feature_columns = featurescount()

    dnn_feature_columns = fixlen_feature_columns
    linear_feature_columns = fixlen_feature_columns

    feature_names = get_feature_names(linear_feature_columns + dnn_feature_columns)

    # 3.generate input data for model
    train, test = train_test_split(data, test_size=0.2)

    train_model_input = {name: train[name] for name in feature_names}
    test_model_input = {name: test[name] for name in feature_names}

    # 4.Define Model,train,predict and evaluate

    device = "cuda:0" if flow.cuda.is_available() else "cpu"

    model = DCN(
        linear_feature_columns=linear_feature_columns,
        dnn_feature_columns=dnn_feature_columns,
        task="binary",
        l2_reg_embedding=1e-5,
        device=device,
    )

    model.compile(
        "adagrad", "binary_crossentropy", metrics=["binary_crossentropy", "auc"],
    )
    model.fit(
        train_model_input,
        train[target].values,
        batch_size=32,
        epochs=10,
        verbose=2,
        validation_split=0.0,
    )

    pred_ans = model.predict(test_model_input, 256)
    print("")
    print("test LogLoss", round(log_loss(test[target].values, pred_ans), 4))
    print("test AUC", round(roc_auc_score(test[target].values, pred_ans), 4))
