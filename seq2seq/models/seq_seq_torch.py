import torch.nn as nn
import torch
import torch.nn.functional as F
from utils.dataset import *
from models.GRU_torch import *
device = 'cuda'


class EncoderRNN_torch(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(EncoderRNN_torch, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(num_embeddings=input_size, embedding_dim=hidden_size)
        self.gru = GRU_pytorch(input_size=hidden_size, hidden_size=hidden_size)

    def forward(self, input, hidden):
        embedded = self.embedding(input).view(1, 1, -1)
        output = embedded
        output, hidden = self.gru(output, hidden)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, self.hidden_size, device=device)


class AttnDecoderRNN_torch(nn.Module):
    '''
    encoder与decoder间的attention计算,
    翻译句子最大长度为MAX_LENGTH
    '''

    def __init__(self, hidden_size, output_size, dropout_p=0.1, max_length=MAX_LENGTH):
        super(AttnDecoderRNN_torch, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = max_length

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = GRU_pytorch(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, hidden, encoder_outputs):
        embedded = self.embedding(input).view(1, 1, -1)
        embedded = self.dropout(embedded)
        attn_weights = F.softmax(
            self.attn(
                torch.cat((embedded[0], hidden), 1)), dim=-1)     # embed和当前的hidden组合
        attn_applied = torch.matmul(attn_weights.unsqueeze(0),
                                 encoder_outputs.unsqueeze(0))

        output = torch.cat((embedded[0], attn_applied[0]), 1)
        output = self.attn_combine(output).unsqueeze(0)

        output = F.relu(output)
        output, hidden = self.gru(output, hidden)
        output = F.log_softmax(self.out(output[0]), dim=1)
        return output, hidden, attn_weights

    def initHidden(self):
        return torch.zeros(1, self.hidden_size, device=device)


