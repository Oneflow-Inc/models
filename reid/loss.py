from __future__ import absolute_import
from __future__ import division

import oneflow.experimental as flow

import numpy as np

class TripletLoss(flow.nn.Module):
    """Triplet loss with hard positive/negative mining.
    
    Reference:
        Hermans et al. In Defense of the Triplet Loss for Person Re-Identification. arXiv:1703.07737.
    
    Imported from `<https://github.com/Cysu/open-reid/blob/master/reid/loss/triplet.py>`_.
    
    Args:
        margin (float, optional): margin for triplet. Default is 0.3.
    """
    
    def __init__(self, margin=0.3):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.ranking_loss = flow.nn.MarginRankingLoss(margin=margin)

    def forward(self, inputs, targets):
        """
        Args:
            inputs (torch.Tensor): feature matrix with shape (batch_size, feat_dim).
            targets (torch.LongTensor): ground truth labels with shape (num_classes).
        """
        n = inputs.size(0)
        
        # Compute pairwise distance, replace by the official when merged
        dist = flow.pow(inputs, 2).sum(dim=1).expand(n, n)
        dist = dist + flow.transpose(dist, dim0= 1, dim1 = 0)
        temp1 = -2*flow.matmul(inputs, flow.transpose(inputs, dim0= 1, dim1 = 0))
        dist = flow.add(dist,temp1)
        dist = flow.sqrt(flow.clamp(dist, min=1e-12))
        # For each anchor, find the hardest positive and negative
        mask = targets.expand(n,n).eq(flow.Tensor(flow.transpose(targets.expand(n,n),dim0= 1, dim1 = 0)))
        dist_ap, dist_an = [], []
        y1 = flow.zeros((1, n),dtype=flow.float32).to('cuda')
        y2 = flow.Tensor(np.exp(100*np.ones((1,n)))).to('cuda')

        for i in range(n):
            temp_dist = flow.slice(dist, [(i, i + 1, 1)])
            temp_mask = flow.slice(mask, [(i, i + 1, 1)])
            temp_mask_rev = flow.slice(1-mask, [(i, i + 1, 1)])
            dist_ap.append(temp_mask.where(temp_dist,y1).max())
            dist_an.append(temp_mask_rev.where(temp_dist,y2).min())

        dist_ap = flow.cat(dist_ap)
        dist_an = flow.cat(dist_an)
        
        # Compute ranking hinge loss
        y = flow.ones_like(dist_an)
        return self.ranking_loss(dist_an, dist_ap, y)


class CrossEntropyLoss(flow.nn.Module):
    r"""Cross entropy loss with label smoothing regularizer.

    Reference:
        Szegedy et al. Rethinking the Inception Architecture for Computer Vision. CVPR 2016.

    With label smoothing, the label :math:`y` for a class is computed by

    .. math::
        \begin{equation}
        (1 - \epsilon) \times y + \frac{\epsilon}{K},
        \end{equation}

    where :math:`K` denotes the number of classes and :math:`\epsilon` is a weight. When
    :math:`\epsilon = 0`, the loss function reduces to the normal cross entropy.

    Args:
        num_classes (int): number of classes.
        epsilon (float, optional): weight. Default is 0.1.
        use_gpu (bool, optional): whether to use gpu devices. Default is True.
        label_smooth (bool, optional): whether to apply label smoothing. Default is True.
    """

    def __init__(self, num_classes, epsilon=0.1):
        super(CrossEntropyLoss, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.logsoftmax = flow.nn.LogSoftmax(dim=1)

    def forward(self, inputs, targets):
        """
        Args:
            inputs (torch.Tensor): prediction matrix (before softmax) with
                shape (batch_size, num_classes).
            targets (torch.LongTensor): ground truth labels with shape (batch_size).
                Each position contains the label index.
        """
        log_probs = self.logsoftmax(inputs)
        targets = flow.tensor(np.eye(self.num_classes)[targets.numpy()], dtype = flow.float32)
        targets = targets.to('cuda')
        targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
        loss = (- targets * log_probs).mean(0).sum()
        return loss
