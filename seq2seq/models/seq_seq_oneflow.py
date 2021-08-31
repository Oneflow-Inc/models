from utils.dataset import *
from utils.utils_oneflow import *
from models.GRU_oneflow import *


class EncoderRNN_oneflow(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(
            num_embeddings=input_size, embedding_dim=hidden_size
        )
        self.gru = GRU_oneflow(input_size=hidden_size, hidden_size=hidden_size)

    def forward(self, input, hidden):
        embedded = self.embedding(input).reshape(1, 1, -1)
        output = embedded
        output, hidden = self.gru(output, hidden)
        return output, hidden

    def init_Hidden(self):
        return flow.zeros((1, self.hidden_size))


class AttnDecoderRNN_oneflow(nn.Module):
    def __init__(self, hidden_size, output_size, dropout_p=0.1, max_length=MAX_LENGTH):
        super().__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = max_length

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = GRU_oneflow(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)
        self.logsoftmax = flow.nn.LogSoftmax(dim=1)

    def forward(self, input, hidden, encoder_outputs):
        embedded = self.embedding(input)
        embedded = self.dropout(embedded)
        attn_weights = flow.softmax(self.attn(flow.cat((embedded[0], hidden), -1)))
        attn_applied = flow.matmul(
            attn_weights.unsqueeze(0), encoder_outputs.unsqueeze(0)
        )
        output = flow.cat((embedded[0], attn_applied[0]), 1)
        output = self.attn_combine(output).unsqueeze(0)
        output = flow.relu(output)
        output, hidden = self.gru(output, hidden)
        output = self.logsoftmax(self.out(output[0]))
        return output, hidden, attn_weights

    def init_Hidden(self):
        return flow.zeros([1, self.hidden_size])
    
