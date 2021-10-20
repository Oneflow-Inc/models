import oneflow.nn as nn
from LSTM import BiLSTM

class Encoder(nn.Module):
    """Applies a multi-layer LSTM to an variable length input sequence.
    """

    def __init__(self, input_size, hidden_size, num_layers,
                 dropout=0.0, bidirectional=True):
        super(Encoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.dropout = dropout
        self.rnn = BiLSTM(input_size, hidden_size, num_layers=num_layers, bi_flag=bool(bidirectional))
    def forward(self, padded_input):
        """
        Args:
            padded_input: N x T x D
            input_lengths: N

        Returns: output, hidden
            - **output**: N x T x H
            - **hidden**: (num_layers * num_directions) x N x H 
        """
        output, hidden = self.rnn(padded_input)
        return output, hidden
