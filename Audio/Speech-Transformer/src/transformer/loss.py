import numpy as np
import oneflow as flow
import oneflow.nn as nn
import oneflow.nn.functional as F

from utils import IGNORE_ID


def cal_performance(pred, gold, smoothing=0.0):
    """Calculate cross entropy loss, apply label smoothing if needed.
    Args:
        pred: N x T x C, score before softmax
        gold: N x T
    """
    smoothing = flow.tensor(smoothing).to(pred.device)
    pred = pred.view(-1, pred.size(2))
    gold = gold.view(-1)
    loss = cal_loss(pred, gold, smoothing)
    pred = flow.tensor(np.argmax(pred.numpy(), axis=1), dtype=flow.int64).to(
        pred.device
    )
    non_pad_mask = gold.ne(IGNORE_ID)
    n_correct = flow.eq(pred, gold)
    n_correct = n_correct.masked_select(non_pad_mask).sum().numpy()

    return loss, n_correct


def cal_loss(pred, gold, smoothing=0.0):
    """Calculate cross entropy loss, apply label smoothing if needed.
    """
    if smoothing > 0.0:
        eps = smoothing
        n_class = pred.size(1)

        # Generate one-hot matrix: N x C.
        # Only label position is 1 and all other positions are 0
        # gold include -1 value (IGNORE_ID) and this will lead to assert error
        gold_for_scatter = gold.ne(IGNORE_ID).long() * gold
        one_hot = F.one_hot(gold_for_scatter, n_class).to(dtype=flow.float32)
        one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / n_class
        sof_prb = F.softmax(pred)
        log_prb = flow.log(sof_prb)

        non_pad_mask = gold.ne(IGNORE_ID)
        n_word = float(non_pad_mask.sum().numpy())
        loss = -(one_hot * log_prb).sum(dim=1)
        loss = loss.masked_select(non_pad_mask).sum() / n_word
    else:
        loss_fn = nn.CrossEntropyLoss(ignore_index=IGNORE_ID).to(pred.device)
        loss = loss_fn(pred, gold)

    return loss
