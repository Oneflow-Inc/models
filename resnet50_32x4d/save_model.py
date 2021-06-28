import torch
import oneflow as flow 
from models.resnext50_32x4d import resnext50_32x4d

parameters = torch.load("resnext50_32x4d-7cdf4587.pth")
new_parameters = dict()
for key,value in parameters.items():
     if "num_batches_tracked" not in key:
          val = value.detach().cpu().numpy()
          new_parameters[key] = val

flow.env.init()
flow.enable_eager_execution()

resnext50_32x4d_module = resnext50_32x4d()
resnext50_32x4d_module.load_state_dict(new_parameters)
flow.save(resnext50_32x4d_module.state_dict(), "resnext50_32x4d_oneflow_model")
