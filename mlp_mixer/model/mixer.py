import random
import numpy as np
from functools import partial

import oneflow
import oneflow as flow
import oneflow.nn as nn
import oneflow.F as F
from utils.drop import DropPath
from layers.mlp import Mlp
from layers.layer_norm import LayerNorm
from layers.helpers import to_2tuple
from layers.patch_embed import PatchEmbed


# TODO: 在测试单个的mlp-block时，出现了nan
# test-code
# test_data = flow.ones((10, 16, 768), dtype=flow.float32)
# block = MixerBlock(768, 16, drop_path=0.5)
# print(block(test_data)) # 有一部分数据输出nan

class MixerBlock(nn.Module):
    def __init__(self, dim, seq_len, mlp_ratio=(0.5, 4.0), mlp_layer=Mlp,
                 norm_layer=partial(LayerNorm, eps=1e-6), act_layer=nn.GELU, drop=0., drop_path=0.):
        super().__init__()
        tokens_dim, channels_dim = [int(x * dim) for x in to_2tuple(mlp_ratio)]
        self.norm1 = norm_layer(dim)
        self.mlp_tokens = mlp_layer(seq_len, tokens_dim, act_layer=act_layer, drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp_channels = mlp_layer(dim, channels_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.mlp_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
        x = x + self.drop_path(self.mlp_channels(self.norm2(x)))
        return x

class MlpMixer(nn.Module):
    def __init__(
            self, 
            num_classes=1000,
            img_size=224,
            in_chans=3,
            patch_size=16,
            num_blocks=8,  # depth
            embed_dim=512,  # hidden dim
            mlp_ratio=(0.5, 4.0),
            block_layer=MixerBlock,
            mlp_layer=Mlp,
            norm_layer=partial(LayerNorm, eps=1e-6),
            act_layer=nn.GELU,
            drop_rate=0.,
            drop_path_rate=0.,
            nlhb=False,  # not used
            stem_norm=False,
        ):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim
        
        self.stem = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans,
            embed_dim=embed_dim, norm_layer=norm_layer if stem_norm else None
        )
        self.blocks = nn.Sequential(*[
            block_layer(
                embed_dim, self.stem.num_patches, mlp_ratio, mlp_layer=mlp_layer, norm_layer=norm_layer,
                act_layer=act_layer, drop=drop_rate, drop_path=drop_path_rate)
            for _ in range(num_blocks)])
        self.norm = norm_layer(embed_dim)
        self.head = nn.Linear(embed_dim, self.num_classes) if num_classes > 0 else nn.Identity()
    
    def forward_features(self, x):
        x = self.stem(x)
        x = self.blocks(x)
        x = self.norm(x)
        x = x.mean(dim=1)
        return x
    
    def forward(self, x):
        print(self.head.weight)
        x = self.forward_features(x)
        x = self.head(x)
        return x

import torch
model_path = "/data/rentianhe/code/vision-mlp-oneflow/weights/torch/mixer_b16_224.pth"

if __name__ == "__main__":
    test_data = flow.ones((1, 3, 224, 224), dtype=flow.float32)
    model = MlpMixer(patch_size=16, num_blocks=12, embed_dim=768)
    preds = model(test_data)
    print(preds)

    # parameters = torch.load(model_path)
    # new_parameters = dict()
    # for key,value in parameters.items():
    #     if "num_batches_tracked" not in key:
    #       val = value.detach().cpu().numpy()
    #       new_parameters[key] = val
    # model.load_state_dict(new_parameters)
    # flow.save(model.state_dict(), "/data/rentianhe/code/vision-mlp-oneflow/weights/flow/mixer_b16_224.of")
    # state_dict = flow.load("/data/rentianhe/code/vision-mlp-oneflow/weights/flow/mixer_b16_224.of")
    # model.load_state_dict(state_dict)