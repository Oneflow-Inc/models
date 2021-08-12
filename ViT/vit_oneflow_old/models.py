import oneflow
import oneflow.F as F
import oneflow as flow
from oneflow import nn
from oneflow import optim
import random
import copy

import numpy as np
import math
import configs as configs

class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."

    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.eps = eps
        self.weight = nn.Parameter(flow.Tensor(flow.ones(features, dtype=flow.float32)))
        self.bias = nn.Parameter(flow.Tensor(flow.zeros(features, dtype=flow.float32)))

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)

        std = x.std(dim=-1, keepdim=True)
        return self.weight * (x - mean) / (std + self.eps) + self.bias

# helpers
def pair(t):
    return t if isinstance(t, tuple) else (t, t)

class Attention(nn.Module):
    def __init__(self, config):
        super(Attention, self).__init__()
        self.num_attention_heads = config.transformer.num_heads
        self.attention_head_size = int(config.hidden_size / self.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.out = nn.Linear(config.hidden_size, config.hidden_size)
        self.attn_dropout = nn.Dropout(config.transformer.attention_dropout_rate)
        self.proj_dropout = nn.Dropout(config.transformer.attention_dropout_rate)

        self.softmax = nn.Softmax(dim=-1)
    
    def transpose_for_scores(self, x):
        # use tuple() instead of directly add flow.Size with tuple
        x = x.reshape(shape=tuple(x.size()[:-1]) + (self.num_attention_heads, self.attention_head_size))
        return x.permute(0, 2, 1, 3)
    
    def forward(self, x):
        # to qkv
        mixed_query = self.query(x)
        mixed_key = self.key(x)
        mixed_value = self.value(x)
        
        # transpose layer
        query_layer = self.transpose_for_scores(mixed_query)
        key_layer = self.transpose_for_scores(mixed_key)
        value_layer = self.transpose_for_scores(mixed_value)

        # count attention
        dots = flow.matmul(query_layer, key_layer.transpose(-1, -2))
        dots = dots / math.sqrt(self.attention_head_size)
        attentions = self.softmax(dots)
        attentions = self.attn_dropout(attentions)

        # weight sum
        context_layer = flow.matmul(attentions, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3)
        new_context_layer_shape = tuple(context_layer.size()[:-2]) + (self.all_head_size, )
        context_layer = context_layer.view(*new_context_layer_shape)

        # merge and projection
        attention_output = self.out(context_layer)
        attention_output = self.proj_dropout(attention_output)
        return attention_output

class Mlp(nn.Module):
    def __init__(self, config):
        super(Mlp, self).__init__()
        self.fc1 = nn.Linear(config.hidden_size, config.transformer.mlp_dim)
        self.fc2 = nn.Linear(config.transformer.mlp_dim, config.hidden_size)
        self.act_fn = nn.GELU()
        self.dropout = nn.Dropout(config.transformer.dropout_rate)

        self._init_weights()
    
    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x

class Embeddings(nn.Module):
    """Construct the embeddings from patch, position embeddings.
    """
    def __init__(self, config, img_size, in_channels=3):
        super(Embeddings, self).__init__()
        img_size = pair(img_size)
        patch_size = pair(config.patches.size)
        n_patches = (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1])
        self.patch_embeddings = nn.Conv2d(
            in_channels = in_channels,
            out_channels = config.hidden_size,
            kernel_size = patch_size,
            stride = patch_size
        )
        self.position_embeddings = nn.Parameter(flow.tensor(np.zeros((1, n_patches+1, config.hidden_size)), dtype=flow.float32))
        self.cls_token = nn.Parameter(flow.tensor(np.zeros((1, 1, config.hidden_size)), dtype=flow.float32))

        self.dropout = nn.Dropout(config.transformer.dropout_rate)
    
    def forward(self, x):
        B = x.shape[0]
        cls_tokens = self.cls_token.expand(B, -1, -1)

        # patch embedding
        x = self.patch_embeddings(x)
        x = x.flatten(2)
        x = x.transpose(-1, -2)
        x = flow.cat((cls_tokens, x), dim=1)

        # add position embeddings
        embeddings = x + self.position_embeddings
        embeddings = self.dropout(embeddings)
        return embeddings

class Block(nn.Module):
    def __init__(self, config):
        super(Block, self).__init__()
        self.hidden_size = config.hidden_size
        self.attention_norm = LayerNorm(config.hidden_size, eps=1e-6)
        self.ffn_norm = LayerNorm(config.hidden_size, eps=1e-6)
        self.ffn = Mlp(config)
        self.attn = Attention(config)

    def forward(self, x):
        # self-attention with pre-norm
        h = x
        x = self.attention_norm(x)
        x = self.attn(x)
        x = x + h
        # ffn with pre-norm
        h = x
        x = self.ffn_norm(x)
        x = self.ffn(x)
        x = x + h
        return x

class Encoder(nn.Module):
    def __init__(self, config):
        super(Encoder, self).__init__()
        self.layer = nn.ModuleList()
        self.encoder_norm = LayerNorm(config.hidden_size, eps=1e-6)
        for _ in range(config.transformer.num_layers):
            layer = Block(config)
            self.layer.append(layer)

    def forward(self, hidden_states):
        for layer_block in self.layer:
            hidden_states = layer_block(hidden_states)
        encoded = self.encoder_norm(hidden_states)
        return encoded

class Transformer(nn.Module):
    def __init__(self, config, img_size):
        super(Transformer, self).__init__()
        self.embeddings = Embeddings(config, img_size=img_size)
        self.encoder = Encoder(config)

    def forward(self, input_ids):
        embedding_output = self.embeddings(input_ids)
        encoded = self.encoder(embedding_output)
        return encoded

class VisionTransformer(nn.Module):
    def __init__(self, config, img_size=224, num_classes=100, zero_head=False, vis=False):
        super(VisionTransformer, self).__init__()
        self.num_classes = num_classes
        self.zero_head = zero_head
        self.classifier = config.classifier

        self.transformer = Transformer(config, img_size)
        self.head = nn.Linear(config.hidden_size, num_classes)

    def forward(self, x):
        x = self.transformer(x)
        logits = self.head(x[:, 0])
        return logits


CONFIGS = {
    'ViT-B_16': configs.get_b16_config(),
    'ViT-B_32': configs.get_b32_config(),
    'ViT-L_16': configs.get_l16_config(),
    'ViT-L_32': configs.get_l32_config(),
    'ViT-H_14': configs.get_h14_config(),
}