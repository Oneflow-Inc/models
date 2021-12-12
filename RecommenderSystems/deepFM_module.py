from collections import OrderedDict

import oneflow as flow
import oneflow.nn as nn

from typing import Any


class Embedding(nn.Embedding):
    def __init__(self, vocab_size, embed_size, split_axis=0):
        # TODO: name and split_axis for weight
        super(Embedding, self).__init__(vocab_size, embed_size,
                                        padding_idx=0)  # padding_idx，说明补长句子的数字是几，后面embedding会映射成0
        for param in self.parameters():
            nn.init.uniform_(param, a=-0.05, b=0.05)  # a~b的均匀分布

    def forward(self, indices):
        # indices = flow.parallel_cast(indices, distribute=flow.distribute.broadcast())
        # print(self.weight) # vocab_size*embed_size
        embedding = flow._C.gather(self.weight, indices, axis=0)
        # print(indices.shape)
        # print(embedding)# indices.shape*embed_size
        return embedding
        # return embedding.view(-1, embedding.shape[-1] * embedding.shape[-2])




class Dense(nn.Module):
    def __init__(
            self, in_features: int, out_features: int, dropout_rate: float = 0.5
    ) -> None:
        super(Dense, self).__init__()
        self.features = nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
        )
        for name, param in self.named_parameters():  # 网络层的名字和参数的迭代器
            if name.endswith("weight"):
                nn.init.xavier_uniform_(param)  # 初始权重用Xavier分布初始化
            elif name.endswith("bias"):  # bias初始化为0
                nn.init.zeros_(param)

    def forward(self, x: flow.Tensor) -> flow.Tensor:
        x = self.features(x)
        return x


class WideAndDeep(nn.Module):
    def __init__(self, FLAGS) -> None:
        super(WideAndDeep, self).__init__()
        self.fm_1st_sparse_embedding = Embedding(vocab_size=FLAGS.wide_vocab_size, embed_size=1)
        self.fm_1st_scores = nn.Linear(FLAGS.num_deep_sparse_fields + FLAGS.num_dense_fields, 1)

        self.deep_embedding = Embedding(
            vocab_size=FLAGS.deep_vocab_size,
            embed_size=FLAGS.deep_embedding_vec_size,
            split_axis=1,
        )

        deep_feature_size = (
                FLAGS.deep_embedding_vec_size * FLAGS.num_deep_sparse_fields
                + FLAGS.num_dense_fields
        )  # 16*26+13
        self.linear_layers = nn.Sequential(
            OrderedDict(  # 有序字典
                [
                    (
                        f"fc{i}",  # 第i层全连接层
                        Dense(
                            deep_feature_size if i == 0 else FLAGS.hidden_size,
                            FLAGS.hidden_size,
                            FLAGS.deep_dropout_rate,
                        ),
                    )
                    for i in range(FLAGS.hidden_units_num)
                ]
            )
        )
        self.deep_scores = nn.Linear(FLAGS.hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(
            self, dense_fields, wide_sparse_fields, deep_sparse_fields
    ) -> flow.Tensor:
        # wide_embedding = self.wide_embedding(wide_sparse_fields)
        # wide_scores = flow.sum(wide_embedding, dim=1, keepdim=True)
        deep_embedding = self.deep_embedding(deep_sparse_fields)
        # FM 1st
        fm_1st_sparse_embedding = self.fm_1st_sparse_embedding(deep_sparse_fields)
        fm_1st_sparse_embedding = fm_1st_sparse_embedding.view(-1, fm_1st_sparse_embedding.shape[-1] *
                                                               fm_1st_sparse_embedding.shape[-2])
        fm_1st_features = flow.cat([fm_1st_sparse_embedding, dense_fields], dim=1)
        fm_1st_scores = self.fm_1st_scores(fm_1st_features)  # shape
        # FM 2nd

        fm_2nd_sparse_embedding = deep_embedding  # [bs, num_deep_sparse_fields, emb_size]

        # square of sum
        sum_embed = flow.sum(fm_2nd_sparse_embedding, 1)  # [bs, emb_size]
        square_sum_embed = sum_embed * sum_embed  # [bs, emb_size]
        # sum of square
        square_embed = fm_2nd_sparse_embedding * fm_2nd_sparse_embedding  # [bs, n, emb_size]
        sum_square_embed = flow.sum(square_embed, 1)  # [bs, emb_size]

        sub = square_sum_embed - sum_square_embed
        sub = sub * 0.5  # [bs, emb_size]
        fm_2nd_scores = flow.sum(sub, 1, keepdim=True)  # [bs, 1]

        # DNN
        # deep_embedding = self.deep_embedding(deep_sparse_fields)
        deep_embedding = deep_embedding.view(-1, deep_embedding.shape[-1] * deep_embedding.shape[-2])
        deep_features = flow.cat([deep_embedding, dense_fields], dim=1)
        deep_features = self.linear_layers(deep_features)
        deep_scores = self.deep_scores(deep_features)
        return self.sigmoid(fm_1st_scores + fm_2nd_scores + deep_scores)


def wide_and_deep(
        pretrained: bool = False, progress: bool = True, **kwargs: Any
) -> WideAndDeep:
    r"""WideAndDeep model architecture from the
    `"One weird trick..." <https://arxiv.org/abs/1606.07792>`_ paper.
    Args:
        pretrained (bool): If True, returns a model pre-trained on WideAndDeep
        progress (bool): If True, displays a progress bar of the download to stderr
     """
    model = WideAndDeep(**kwargs)
    return model


if __name__ == "__main__":
    ebd = Embedding(vocab_size=10, embed_size=3)
    a = ebd(flow.tensor([[1, 2, 3], [2, 3, 4]]))
    print(a.shape)