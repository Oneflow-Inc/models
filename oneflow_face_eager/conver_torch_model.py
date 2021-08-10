import torch

from backbones import get_model
import oneflow as flow

backbone = get_model("r100")
parameters = torch.load("backbone.pth")
new_parameters = dict()
for key,value in parameters.items():
     if "num_batches_tracked" not in key:
          val = value.detach().cpu().numpy()
          new_parameters[key] = val
backbone.load_state_dict(new_parameters)
flow.save(backbone.state_dict(), "oneflow_face_PReLU")