import oneflow.nn as nn

from model.transformer import TransformerBlock
from model.embedding.bert import BERTEmbedding
import numpy as np
import oneflow as flow


class BERT(nn.Module):
    """
    BERT model : Bidirectional Encoder Representations from Transformers.
    """

    def __init__(self, vocab_size, hidden=768, n_layers=12, attn_heads=12, dropout=0.1):
        """
        :param vocab_size: vocab_size of total words
        :param hidden: BERT model hidden size
        :param n_layers: numbers of Transformer blocks(layers)
        :param attn_heads: number of attention heads
        :param dropout: dropout rate
        """

        super().__init__()
        self.hidden = hidden
        self.n_layers = n_layers
        self.attn_heads = attn_heads

        # paper noted they used 4*hidden_size for ff_network_hidden_size
        self.feed_forward_hidden = hidden * 4

        # embedding for BERT, sum of positional, segment, token embeddings
        self.embedding = BERTEmbedding(
            vocab_size=vocab_size, embed_size=hidden)

        # multi-layers transformer blocks, deep network
        self.transformer_blocks = nn.ModuleList(
            [
                TransformerBlock(hidden, attn_heads, hidden * 4, dropout)
                for _ in range(n_layers)
            ]
        )

    # x.shape >> flow.Size([16, 20])
    def forward(self, x: flow.Tensor, mask: flow.Tensor, segment_info: flow.Tensor) -> flow.Tensor:
        """[summary]

        Args:
            x (flow.Tensor): sentence input with shape [batchsize, seq]
            mask (flow.Tensor): mask input with shape [batchsize, seq]
            segment_info (flow.Tensor): token type with shape [batchsize, seq]
        """
        # mask = (
        #     (x > 0)
        #     .unsqueeze(1)
        #     .repeat((1, x.shape[1], 1))
        #     .unsqueeze(1)
        #     .repeat((1, 8, 1, 1))
        # )

        # attention masking for padded token
        mask = mask.unsqueeze(1).repeat(
            (1, x.shape[1], 1)).unsqueeze(1).repeat((1, 8, 1, 1))

        # embedding the indexed sequence to sequence of vectors
        x = self.embedding(x, segment_info)

        # running over multiple transformer blocks
        for transformer in self.transformer_blocks:
            x = transformer(x, mask)

        return x
