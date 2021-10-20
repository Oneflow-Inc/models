import oneflow as flow
import oneflow.nn as nn


class DotProductAttention(nn.Module):
    r"""Dot product attention.
    Given a set of vector values, and a vector query, attention is a technique
    to compute a weighted sum of the values, dependent on the query.

    NOTE: Here we use the terminology in Stanford cs224n-2018-lecture11.
    """

    def __init__(self):
        super(DotProductAttention, self).__init__()
        # TODO: move this out of this class?

    def forward(self, queries, values):
        """
        # rnn_output.unsqueeze(dim=1): torch.Size([2,1,512])
        # encoder_padded_outputs: torch.Size([2, 587,512])
        Args:
            queries: N x To x H
            values : N x Ti x H

        Returns:
            output: N x To x H
            attention_distribution: N x To x Ti
        """
        batch_size = queries.size(0)
        hidden_size = queries.size(2)
        input_lengths = values.size(1)
        attention_scores = flow.bmm(queries, values.transpose(1, 2))
        attention_distribution = nn.Softmax(dim=1)(attention_scores.view(-1, input_lengths)).view(batch_size, -1, input_lengths)
        attention_output = flow.bmm(attention_distribution, values)

        return attention_output, attention_distribution
