from .lstm_oneflow import LSTM
import oneflow.nn as nn


class lstm_ser(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, batch_size):
        super(lstm_ser, self).__init__()
        # self.lstm_feature = LSTM(input_dim, hidden_dim, hidden_dim)
        self.classifier = nn.Sequential(
            LSTM(input_dim, hidden_dim, batch_size),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, output_dim),
        )

    def forward(self, x):
        # x = self.lstm_feature(x)

        logits = self.classifier(x)
        return logits


if __name__ == "__main__":
    model = lstm_ser(312, 256, 10, 32)
    model.to("cuda")
    arr = np.random.randn(1, 32, 312)
    input = flow.Tensor(arr, device="cuda")
    output = model(input)

    print(output.shape)
