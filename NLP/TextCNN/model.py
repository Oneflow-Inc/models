import oneflow as flow
from oneflow import nn
from oneflow.nn.parameter import Parameter

#%% Text CNN model
class textCNN(nn.Module):
    def __init__(
        self,
        word_emb_dim,
        vocab_size,
        dim_channel,
        kernel_wins,
        dropout_rate,
        num_class,
        max_seq_len,
        training=True,
    ):
        super(textCNN, self).__init__()
        self.embed = nn.Embedding(vocab_size, word_emb_dim)
        self.convs = nn.ModuleList(
            [nn.Conv2d(1, dim_channel, (w, word_emb_dim)) for w in kernel_wins]
        )
        self.maxpool = nn.ModuleList(
            [nn.MaxPool2d((max_seq_len - w + 1, 1), stride=1) for w in kernel_wins]
        )
        # Dropout layer
        self.dropout = nn.Dropout(dropout_rate)
        self.training = training
        # FC layer
        self.fc = nn.Linear(len(kernel_wins) * dim_channel, num_class)

    def forward(self, x):
        emb_x = self.embed(x)
        emb_x = emb_x.unsqueeze(1)
        con_x = [conv(emb_x) for conv in self.convs]
        pool_x = []
        for i in range(len(con_x)):
            cur_maxpool_layer = self.maxpool[i]
            pool_x.append(cur_maxpool_layer(con_x[i]).squeeze(-1).squeeze(-1))
        fc_x = flow.cat(pool_x, dim=1)
        if self.training:
            fc_x = self.dropout(fc_x)
        logit = self.fc(fc_x)
        return logit
