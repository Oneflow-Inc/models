import warnings
from typing import List, Optional
import oneflow as flow
from oneflow import nn, Tensor

class FrozenBatchNorm2d(nn.Module):
    """
    BatchNorm2d where the batch statistics and the affine parameters
    are fixed
    """

    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        n: Optional[int] = None,
    ):
        # n=None for backward-compatibility
        if n is not None:
            warnings.warn("`n` argument is deprecated and has been renamed `num_features`",
                          DeprecationWarning)
            num_features = n
        super(FrozenBatchNorm2d, self).__init__()
        self.eps = eps
        self.register_buffer("weight", flow.Tensor(flow.ones(num_features)))
        self.register_buffer("bias", flow.Tensor(flow.zeros(num_features)))
        self.register_buffer("running_mean", flow.Tensor(flow.zeros(num_features)))
        self.register_buffer("running_var", flow.Tensor(flow.ones(num_features)))

    def _load_from_state_dict(
        self,
        state_dict: dict,
        prefix: str,
        local_metadata: dict,
        strict: bool,
        missing_keys: List[str],
        unexpected_keys: List[str],
        error_msgs: List[str],
    ):
        num_batches_tracked_key = prefix + 'num_batches_tracked'
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]

        super(FrozenBatchNorm2d, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs)

    def forward(self, x: Tensor) -> Tensor:
        # move reshapes to the beginning
        # to make it fuser-friendly
        w = self.weight.reshape([1, -1, 1, 1])
        b = self.bias.reshape([1, -1, 1, 1])
        rv = self.running_var.reshape([1, -1, 1, 1])
        rm = self.running_mean.reshape([1, -1, 1, 1])
        scale = w * (rv + self.eps).rsqrt()
        bias = b - rm * scale
        return x * scale + bias

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.weight.shape[0]}, eps={self.eps})"