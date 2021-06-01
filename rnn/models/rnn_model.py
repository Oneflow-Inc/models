import oneflow.experimental as flow
import oneflow.experimental.nn as nn

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()

        self.hidden_size = hidden_size

        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(input_size + hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

        # TODO(Liang Depeng): oneflow does not support `flow.cat` yet
        self.cat = (
            flow.builtin_op("concat")
                .Input("in", 2)
                .Attr("axis", 1)
                .Attr("max_dim_size", input_size + hidden_size)
                .Output("out")
                .Build()
        )

    def forward(self, input, hidden):
        # NOTE(Liang Depeng): original torch implementation 
        # combined = torch.cat((input, hidden), 1)
        combined = self.cat(input, hidden)[0]
        hidden = self.i2h(combined)
        output = self.i2o(combined)
        #output = self.softmax(output)
        return output, hidden

    def initHidden(self):
        # NOTE(Liang Depeng): original torch implementation 
        # return torch.zeros(1, self.hidden_size)
        hidden = flow.Tensor(1, self.hidden_size)
        flow.nn.init.zeros_(hidden)
        return hidden.to("cuda")
