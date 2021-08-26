import torch
import oneflow as flow
import numpy as np
from model.mnasnet import mnasnet0_5, mnasnet1_0

def load_and_save(model, checkpoint_path):
    state_dict = torch.load(checkpoint_path)
    new_parameters = dict()
    for key, value in state_dict.items():
        if "num_batches_tracked" not in key:
            val = value.detach().cpu().numpy()
            new_parameters[key] = val.astype(np.float32)
    model.load_state_dict(new_parameters)
    flow.save(model.state_dict(), "/data/rentianhe/code/new_models/models/mnasnet/weight/flow/mnasnet1_0")

if __name__ == "__main__":
    mnasnet0_5_weight = "/data/rentianhe/code/new_models/models/mnasnet/weight/torch/mnasnet0.5_top1_67.823-3ffadce67e.pth"
    mnasnet1_0_weight = "/data/rentianhe/code/new_models/models/mnasnet/weight/torch/mnasnet1.0_top1_73.512-f206786ef8.pth"
    # model = mnasnet0_5()
    model = mnasnet1_0()
    load_and_save(model, mnasnet1_0_weight)