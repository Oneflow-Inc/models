import oneflow as flow
import oneflow.nn as nn
import oneflow.nn.functional as F
import numpy as np

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

class PositionEmbs(nn.Module):
    def __init__(self, num_patches, emb_dim, dropout_rate=0.1):
        super(PositionEmbs, self).__init__()
        self.pos_embedding = nn.Parameter(flow.tensor(np.random.randn(1, num_patches + 1, emb_dim), dtype=flow.float32))
        if dropout_rate > 0:
            self.dropout = nn.Dropout(dropout_rate)
        else:
            self.dropout = None

    def forward(self, x):
        out = x + self.pos_embedding

        if self.dropout:
            out = self.dropout(out)

        return out

class MlpBlock(nn.Module):
    """ Transformer Feed-Forward Block """
    def __init__(self, in_dim, mlp_dim, out_dim, dropout_rate=0.1):
        super(MlpBlock, self).__init__()

        # init layers
        self.fc1 = nn.Linear(in_dim, mlp_dim)
        self.fc2 = nn.Linear(mlp_dim, out_dim)
        self.act = nn.GELU()
        if dropout_rate > 0.0:
            self.dropout1 = nn.Dropout(dropout_rate)
            self.dropout2 = nn.Dropout(dropout_rate)
        else:
            self.dropout1 = None
            self.dropout2 = None

    def forward(self, x):

        out = self.fc1(x)
        out = self.act(out)
        if self.dropout1:
            out = self.dropout1(out)

        out = self.fc2(out)
        if self.dropout2:
            out = self.dropout2(out)
        return out

class SelfAttention(nn.Module):
    def __init__(self, in_dim, heads=8, dropout_rate=0.1):
        super(SelfAttention, self).__init__()
        self.heads = heads
        self.head_dim = in_dim // heads
        self.scale = self.head_dim ** 0.5


        self.query = nn.Linear(in_dim, self.heads * self.head_dim)
        self.key = nn.Linear(in_dim, self.heads * self.head_dim)
        self.value = nn.Linear(in_dim, self.heads * self.head_dim)
        self.out = nn.Linear(self.heads * self.head_dim, in_dim)

        if dropout_rate > 0:
            self.dropout = nn.Dropout(dropout_rate)
        else:
            self.dropout = None

    def transpose_for_scores(self, x):
        B, token_nums, _ = x.size()
        x = x.view(B, token_nums, self.heads, self.head_dim)
        return x.permute(0, 2, 1, 3)

    def forward(self, x):
        b, n, _ = x.shape
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)

        q = self.transpose_for_scores(q)
        k = self.transpose_for_scores(k)
        v = self.transpose_for_scores(v)

        attn_weights = flow.matmul(q, k.transpose(-2, -1)) / self.scale
        attn_weights = nn.Softmax(dim=-1)(attn_weights)
        out = flow.matmul(attn_weights, v)
        out = out.permute(0, 2, 1, 3)
        new_out_shape = tuple(out.size()[:-2]) + (self.heads * self.head_dim, )
        out = out.view(*new_out_shape)
        out = self.out(out)

        return out


class EncoderBlock(nn.Module):
    def __init__(self, in_dim, mlp_dim, num_heads, dropout_rate=0.1, attn_dropout_rate=0.1):
        super(EncoderBlock, self).__init__()

        self.norm1 = LayerNorm(in_dim)
        self.attn = SelfAttention(in_dim, heads=num_heads, dropout_rate=attn_dropout_rate)
        if dropout_rate > 0:
            self.dropout = nn.Dropout(dropout_rate)
        else:
            self.dropout = None
        self.norm2 = LayerNorm(in_dim)
        self.mlp = MlpBlock(in_dim, mlp_dim, in_dim, dropout_rate)

    def forward(self, x):
        residual = x
        out = self.norm1(x)
        out = self.attn(out)
        if self.dropout:
            out = self.dropout(out)
        out += residual
        residual = out

        out = self.norm2(out)
        out = self.mlp(out)
        out += residual
        return out

class Encoder(nn.Module):
    def __init__(self, num_patches, emb_dim, mlp_dim, num_layers=12, num_heads=12, dropout_rate=0.1, attn_dropout_rate=0.0):
        super(Encoder, self).__init__()

        # positional embedding
        self.pos_embedding = PositionEmbs(num_patches, emb_dim, dropout_rate)

        # encoder blocks
        in_dim = emb_dim
        self.encoder_layers = nn.ModuleList()
        for i in range(num_layers):
            layer = EncoderBlock(in_dim, mlp_dim, num_heads, dropout_rate, attn_dropout_rate)
            self.encoder_layers.append(layer)
        self.norm = LayerNorm(in_dim)

    def forward(self, x):

        out = self.pos_embedding(x)

        for layer in self.encoder_layers:
            out = layer(out)

        out = self.norm(out)
        return out

class VisionTransformer(nn.Module):
    """ Vision Transformer """
    def __init__(self,
                 image_size=(256, 256),
                 patch_size=(16, 16),
                 emb_dim=768,
                 mlp_dim=3072,
                 num_heads=12,
                 num_layers=12,
                 num_classes=1000,
                 attn_dropout_rate=0.0,
                 dropout_rate=0.1,
                 feat_dim=None):
        super(VisionTransformer, self).__init__()
        h, w = image_size
        # embedding layer
        fh, fw = patch_size
        gh, gw = h // fh, w // fw
        num_patches = gh * gw
        self.embedding = nn.Conv2d(3, emb_dim, kernel_size=(fh, fw), stride=(fh, fw))
        # class token
        self.cls_token = nn.Parameter(flow.zeros(1, 1, emb_dim))

        # transformer
        self.transformer = Encoder(
            num_patches=num_patches,
            emb_dim=emb_dim,
            mlp_dim=mlp_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout_rate=dropout_rate,
            attn_dropout_rate=attn_dropout_rate)

        # classfier
        self.classifier = nn.Linear(emb_dim, num_classes)

    def forward(self, x):
        emb = self.embedding(x)     # (n, c, gh, gw)
        emb = emb.permute(0, 2, 3, 1)  # (n, gh, hw, c)
        b, h, w, c = emb.shape
        emb = emb.view(b, h * w, c)

        # prepend class token
        cls_token = self.cls_token.repeat(b, 1, 1)
        emb = flow.cat([cls_token, emb], dim=1)

        # transformer
        feat = self.transformer(emb)

        # classifier
        logits = self.classifier(feat[:, 0])
        return logits


def ViT_B_16_224():
    return VisionTransformer(
        image_size=(224, 224),
        patch_size=(16, 16),
        emb_dim=768,
        mlp_dim=3072,
        num_heads=12,
        num_layers=12,
        num_classes=1000,
        attn_dropout_rate=0.,
        dropout_rate=0.1,
        feat_dim=None
    )

def ViT_B_16_384():
    return VisionTransformer(
        image_size=(384, 384),
        patch_size=(16, 16),
        emb_dim=768,
        mlp_dim=3072,
        num_heads=12,
        num_layers=12,
        num_classes=1000,
        attn_dropout_rate=0.,
        dropout_rate=0.1,
        feat_dim=None
    )

def ViT_B_32_224():
    return VisionTransformer(
        image_size=(224, 224),
        patch_size=(32, 32),
        emb_dim=768,
        mlp_dim=3072,
        num_heads=12,
        num_layers=12,
        num_classes=1000,
        attn_dropout_rate=0.,
        dropout_rate=0.1,
        feat_dim=None
    )

def ViT_B_32_384():
    return VisionTransformer(
        image_size=(384, 384),
        patch_size=(32, 32),
        emb_dim=768,
        mlp_dim=3072,
        num_heads=12,
        num_layers=12,
        num_classes=1000,
        attn_dropout_rate=0.,
        dropout_rate=0.1,
        feat_dim=None
    )

def ViT_L_16_384():
    return VisionTransformer(
        image_size=(384, 384),
        patch_size=(16, 16),
        emb_dim=1024,
        mlp_dim=4096,
        num_heads=16,
        num_layers=24,
        num_classes=1000,
        attn_dropout_rate=0.,
        dropout_rate=0.1,
        feat_dim=None
    )

def ViT_L_32_384():
    return VisionTransformer(
        image_size=(384, 384),
        patch_size=(32, 32),
        emb_dim=1024,
        mlp_dim=4096,
        num_heads=16,
        num_layers=24,
        num_classes=1000,
        attn_dropout_rate=0.,
        dropout_rate=0.1,
        feat_dim=None
    )