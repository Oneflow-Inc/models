import oneflow as flow
import torch
from vit_oneflow.model import VisionTransformer as vit_oneflow
from vit_oneflow.checkpoint import convert_torch_to_oneflow as flow_load
from vit_pytorch.model_new import VisionTransformer as vit_torch
from vit_pytorch.checkpoint_torch import load_checkpoint as torch_load
import numpy as np

def main():

    checkpoint_path = "/data/rentianhe/code/ViT-OneFlow/models/ViT/weights/checkpoint/imagenet21k+imagenet2012_ViT-B_16.pth"

    # create model
    flow_model = vit_oneflow(
             image_size=(384, 384),
             patch_size=(16, 16),
             emb_dim=768,
             mlp_dim=3072,
             num_heads=12,
             num_layers=12,
             num_classes=1000,
             attn_dropout_rate=0.0,
             dropout_rate=0.1)
    
    torch_model = vit_torch(
             image_size=(384, 384),
             patch_size=(16, 16),
             emb_dim=768,
             mlp_dim=3072,
             num_heads=12,
             num_layers=12,
             num_classes=1000,
             attn_dropout_rate=0.0,
             dropout_rate=0.1)
    
    # load checkpoint
    if checkpoint_path:
        flow_state_dict = flow_load(checkpoint_path)
        flow_model.load_state_dict(flow_state_dict, strict=False)
        print("Load pretrained weights from {}".format(checkpoint_path))

        torch_state_dict = torch_load(checkpoint_path)
        torch_model.load_state_dict(torch_state_dict, strict=False)
        print("Load pretrained weights from {}".format(checkpoint_path))
    
    test_data = np.ones((1, 3, 384, 384)).astype(np.float32)

    flow_data = flow.tensor(test_data).to("cuda")
    torch_data = torch.tensor(test_data).to("cuda")
    # flow_data = flow.tensor(test_data)
    # torch_data = torch.tensor(test_data)
    flow_model = flow_model.to("cuda")
    torch_model = torch_model.to("cuda")

    flow_model.eval()
    torch_model.eval()

    print(flow_model(flow_data))
    print(torch_model(torch_data))

if __name__ == '__main__':
    main()