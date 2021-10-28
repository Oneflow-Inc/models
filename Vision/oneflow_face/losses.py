import oneflow
from oneflow import nn
import numpy as np

def get_loss(name):
    if name == "cosface":
        return CosFace()
    elif name == "arcface":
        return ArcFace()
    else:
        raise ValueError()


class CosFace(nn.Module):
    def __init__(self, s=64.0, m=0.40):
        super(CosFace, self).__init__()
        self.s = s
        self.m = m

    def forward(self, cosine, label):
        index = oneflow.where(label != -1)[0]
        m_hot = oneflow.zeros(index.size()[0], cosine.size()[1], device=cosine.device) 

        m_hot=oneflow.scatter(m_hot,1, label[index, None], self.m)
        cosine=cosine[index] - m_hot

        ret = cosine * self.s
        return ret


class ArcFace(nn.Module):
    def __init__(self, s=64.0, m=0.5):
        super(ArcFace, self).__init__()
        self.s = s
        self.m = m

    def forward(self, cosine: oneflow.Tensor, label):
        index = oneflow.where(label != -1)[0]
        m_hot = oneflow.zeros(index.size()[0], cosine.size()[1], device=cosine.device)
        m_hot.scatter_(1, label[index, None], self.m)
        cosine.acos_()
        cosine[index] += m_hot
        cosine.cos_().mul_(self.s)
        return cosine
