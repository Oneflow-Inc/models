import os
import torch
import oneflow as flow
from pytorch_mnasnet import mnasnet0_5 as torch_mnasnet0_5
from oneflow_mnasnet import mnasnet0_5 as of_mnasnet0_5

pytorch_model = torch_mnasnet0_5()
oneflow_model = of_mnasnet0_5()

torch_state_dict = pytorch_model.state_dict()

new_parameters = dict()
for k, v in torch_state_dict.items():
    if "num_batches_tracked" not in k:
        new_parameters[k] = flow.tensor(torch_state_dict[k].detach().numpy())

if not os.path.exists("./mnasnet0_5"):
    flow.save(new_parameters, "./mnasnet0_5")

params = flow.load("./mnasnet0_5")
oneflow_model.load_state_dict(params)