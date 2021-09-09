from collections import OrderedDict

import oneflow as flow
import oneflow.nn as nn

from typing import Any


__all__ = ["WideAndDeep", "wide_and_deep"]


class Embedding(nn.Embedding):
    def __init__(self, vocab_size, embed_size, split_axis=0):
        # TODO: name and split_axis for weight
        super(Embedding, self).__init__(vocab_size, embed_size, padding_idx=0)
        for param in self.parameters():
            nn.init.uniform_(param, a=-0.05, b=0.05)

    def forward(self, indices):
        # indices = flow.parallel_cast(indices, distribute=flow.distribute.broadcast())
        embedding = flow._C.gather(self.weight, indices, axis=0)
        return embedding.view(-1, embedding.shape[-1] * embedding.shape[-2])


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
        for name, param in self.named_parameters():
            if name.endswith("weight"):
                nn.init.xavier_uniform_(param)
            elif name.endswith("bias"):
                nn.init.zeros_(param)

    def forward(self, x: flow.Tensor) -> flow.Tensor:
        x = self.features(x)
        return x


class WideAndDeep(nn.Module):
    def __init__(self, FLAGS) -> None:
        super(WideAndDeep, self).__init__()
        self.FLAGS=FLAGS

        self.wide_embedding = Embedding(vocab_size=FLAGS.wide_vocab_size, embed_size=1)
        self.deep_embedding = Embedding(
            vocab_size=FLAGS.deep_vocab_size,
            embed_size=FLAGS.deep_embedding_vec_size,
            split_axis=1,
        )
        deep_feature_size = (
            FLAGS.deep_embedding_vec_size * FLAGS.num_deep_sparse_fields
            + FLAGS.num_dense_fields
        )
        self.linear_layers = nn.Sequential(
            OrderedDict(
                [
                    (
                        f"fc{i}",
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
        if self.FLAGS.mode=='dmp':
            world_size = flow.env.get_world_size()
            placement = flow.placement("cpu", {0: range(world_size)})
            self.world_size=world_size
            self.placement=placement
            self.wide_embedding=self.wide_embedding.to_consistent(placement=placement,sbp=flow.sbp.split(0))
            self.deep_embedding=self.deep_embedding.to_consistent(placement=placement,sbp=flow.sbp.split(0))
            self.linear_layers=self.linear_layers.to_consistent(placement=placement,sbp=flow.sbp.broadcast)
            self.deep_scores=self.deep_scores.to_consistent(placement=placement,sbp=flow.sbp.broadcast)

    def forward(
        self, dense_fields, wide_sparse_fields, deep_sparse_fields
    ) -> flow.Tensor:
        if self.FLAGS.mode=='dmp':
            #print('输入数据：wide_sparse_fields',wide_sparse_fields.sbp)
            wide_embedding = self.wide_embedding(wide_sparse_fields)
            #print('查表gather： wide_embedding',wide_embedding.sbp)
            wide_scores = flow.sum(wide_embedding, dim=1, keepdim=True)
            #print('wide部分score： wide_scores',wide_scores.sbp)

            #最后，为了支持后续的数据并行，因此我们通过插入parallel_cast，将 wide_scores 转变为 Split(0),然后还要to gpu。
            #Cast a local tensor to consistent tensor or cast a consistent tensor to another consistent tensor with different sbp or placement
            wide_scores=wide_scores.to_consistent(placement=self.placement,sbp=flow.sbp.split(0)).to("cuda")

            #print('输入数据：deep_sparse_fields',deep_sparse_fields.sbp)
            deep_embedding = self.deep_embedding(deep_sparse_fields)
        

            #print('查表gather： deep_embedding',deep_embedding.sbp)
            #print('输入数据： dense_fields',dense_fields.sbp)
            deep_features = flow.cat([deep_embedding, dense_fields], dim=1)

            #最后，为了支持后续的数据并行，通过插入to_consistent，将 deep_features（原版是deep_embedding） 转变为 Split(0),然后还要to gpu
            deep_features=deep_features.to_consistent(placement=self.placement,sbp=flow.sbp.split(0)).to("cuda")

            #由于输入dnn的deep_features是split(0),dnn是broadcast，后面的输出都是split(0)
            #print('cat操作后： deep_features',deep_features.sbp)
            deep_features = self.linear_layers(deep_features)
            #print('喂给dnn层后：deep_features', deep_features)
            deep_scores = self.deep_scores(deep_features)
            #print('喂给全连接层后：deep_scores', deep_scores)
            predicts=self.sigmoid(wide_scores + deep_scores)  #predicts是split(0)

            #由于外面的dataloader是broadcast的，所以要把predicts也转为broadcast，方便外边计算metrics
            predicts=predicts.to_consistent(placement=self.placement,sbp=flow.sbp.broadcast).to("cuda")
            return predicts
        else:
            wide_embedding = self.wide_embedding(wide_sparse_fields)
            wide_scores = flow.sum(wide_embedding, dim=1, keepdim=True)

            deep_embedding = self.deep_embedding(deep_sparse_fields)
            deep_features = flow.cat([deep_embedding, dense_fields], dim=1)
            deep_features = self.linear_layers(deep_features)
            deep_scores = self.deep_scores(deep_features)
            return self.sigmoid(wide_scores + deep_scores)


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
