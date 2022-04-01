import numpy as np
import oneflow as flow
import oneflow.nn as nn
from sklearn.metrics import *

class Dice(nn.Module):
    """The Data Adaptive Activation Function in DIN,which can be viewed as a generalization of PReLu and can adaptively adjust the rectified point according to distribution of input data.
    Input shape:
        - 2 dims: [batch_size, embedding_size(features)]
        - 3 dims: [batch_size, num_features, embedding_size(features)]
    Output shape:
        - Same shape as input.
    References
        - [Zhou G, Zhu X, Song C, et al. Deep interest network for click-through rate prediction[C]//Proceedings of the 24th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining. ACM, 2018: 1059-1068.](https://arxiv.org/pdf/1706.06978.pdf)
        - https://github.com/zhougr1993/DeepInterestNetwork, https://github.com/fanoping/DIN-pytorch
    """

    def __init__(self, emb_size, dim=2, epsilon=1e-8, device="cpu"):
        super(Dice, self).__init__()
        assert dim == 2 or dim == 3

        self.bn = nn.BatchNorm1d(emb_size, eps=epsilon)
        self.sigmoid = nn.Sigmoid()
        self.dim = dim

        # wrap alpha in nn.Parameter to make it trainable
        if self.dim == 2:
            self.alpha = nn.Parameter(flow.zeros((emb_size,)).to(device))
        else:
            self.alpha = nn.Parameter(flow.zeros((emb_size, 1)).to(device))

    def forward(self, x):
        assert x.dim() == self.dim
        if self.dim == 2:
            x_p = self.sigmoid(self.bn(x))
            out = self.alpha * (1 - x_p) * x + x_p * x
        else:
            x = flow.transpose(x, 1, 2)
            x_p = self.sigmoid(self.bn(x))
            out = self.alpha * (1 - x_p) * x + x_p * x
            out = flow.transpose(out, 1, 2)
        return out


class Identity(nn.Module):
    def __init__(self, **kwargs):
        super(Identity, self).__init__()

    def forward(self, inputs):
        return inputs


def activation_layer(act_name, hidden_size=None, dice_dim=2):
    """Construct activation layers
    Args:
        act_name: str or nn.Module, name of activation function
        hidden_size: int, used for Dice activation
        dice_dim: int, used for Dice activation
    Return:
        act_layer: activation layer
    """
    if isinstance(act_name, str):
        if act_name.lower() == "sigmoid":
            act_layer = nn.Sigmoid()
        elif act_name.lower() == "linear":
            act_layer = Identity()
        elif act_name.lower() == "relu":
            act_layer = nn.ReLU(inplace=True)
        elif act_name.lower() == "dice":
            assert dice_dim
            act_layer = Dice(hidden_size, dice_dim)
        elif act_name.lower() == "prelu":
            act_layer = nn.PReLU()
    elif issubclass(act_name, nn.Module):
        act_layer = act_name()
    else:
        raise NotImplementedError

    return act_layer



def get_optim(model,optimizer, lr=0.01):
    if isinstance(optimizer, str):
        if optimizer == "sgd":
            optim = flow.optim.SGD(model.parameters(), lr=lr)
        elif optimizer == "adam":
            optim = flow.optim.Adam(model.parameters(), lr=lr)
        elif optimizer == "adagrad":
            optim = flow.optim.Adagrad(model.parameters(), lr=lr)
        elif optimizer == "rmsprop":
            optim = flow.optim.RMSprop(model.parameters(),lr=lr)
        else:
            raise NotImplementedError
    else:
        optim = optimizer
    return optim

def get_loss_func(loss):
    if isinstance(loss, str):
        if loss == "binary_crossentropy":
            loss_func = nn.BCELoss(reduction="sum")
        elif loss == "mse":
            loss_func = nn.MSELoss(reduction="sum")
        elif loss == "mae":
            loss_func = nn.L1Loss(reduction="sum")
        else:
            raise NotImplementedError
    else:
        loss_func = loss
    return loss_func


def get_log_loss(y_true, y_pred, eps=1e-7, normalize=True, sample_weight=None, labels=None):
    # change eps to improve calculation accuracy
    return log_loss(y_true,
                    y_pred,
                    eps,
                    normalize,
                    sample_weight,
                    labels)

def get_metrics(metrics_names, metrics, set_eps=False):
    metrics_ = {}
    metrics_names = metrics_names
    if metrics:
        for metric in metrics:
            if metric == "binary_crossentropy" or metric == "logloss":
                if set_eps:
                    metrics_[metric] = get_log_loss
                else:
                    metrics_[metric] = log_loss
            if metric == "auc":
                metrics_[metric] = roc_auc_score
            if metric == "mse":
                metrics_[metric] = mean_squared_error
            if metric == "accuracy" or metric == "acc":
                metrics_[metric] = lambda y_true, y_pred: accuracy_score(
                    y_true, np.where(y_pred > 0.5, 1, 0))
            metrics_names.append(metric)
    return metrics_, metrics_names