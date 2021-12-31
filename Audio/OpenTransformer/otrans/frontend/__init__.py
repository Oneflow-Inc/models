import oneflow as flow
import oneflow.nn as nn
from .concat import *
from .conv import ConvFrontEnd
from otrans.frontend.base import BaseFrontEnd


BuildFrontEnd = {
    "concat": ConcatFeatureFrontEnd,
    "concat-with-linear": ConcatWithLinearFrontEnd,
    "conv": ConvFrontEnd,
}
