import oneflow as flow
import os
from oneflow import nn


def save_model(
    module: nn.Module, checkpoint_path: str, epoch: int, acc: float, is_consistent: bool
):
    state_dict = module.state_dict()
    save_path = os.path.join(checkpoint_path, "epoch_%d_val_acc_%f" % (epoch, acc))
    if flow.env.get_rank() == 0:
        print(f"Saving model to {save_path}")
    if is_consistent:
        flow.save(state_dict, save_path, consistent_dst_rank=0)
    elif flow.env.get_rank() == 0:
        flow.save(state_dict, save_path)
    else:
        return
