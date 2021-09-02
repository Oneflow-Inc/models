import oneflow.nn as nn
import numpy as np
import oneflow as flow


class cnn1d_ser(nn.Module):
    def __init__(self, in_channels=1, out_channels=32, input_dim=312, hidden_dim=32, output_dim=10):
        super(cnn1d_ser, self).__init__()
        self.classifier = nn.Sequential(nn.Conv1d(in_channels, out_channels, 5, stride=1, padding=2),
                                        nn.BatchNorm1d(out_channels),
                                        nn.ReLU(),
                                        nn.Dropout(0.5),
                                        nn.Conv1d(
                                            out_channels, out_channels, 5, stride=1, padding=2),
                                        nn.BatchNorm1d(out_channels),
                                        nn.ReLU(),
                                        nn.Dropout(0.5),
                                        nn.Flatten(),
                                        nn.Linear(
                                            input_dim * out_channels, hidden_dim),
                                        nn.BatchNorm1d(hidden_dim),
                                        nn.ReLU(),
                                        nn.Dropout(0.5),
                                        nn.Linear(hidden_dim, output_dim)
                                        )

    def forward(self, x):
        x = x.reshape(x.shape[1], 1, x.shape[2])
        x = self.classifier(x)
        return x


if __name__ == '__main__':
    model = cnn1d_ser()
    # model = nn.Conv1d(16, 33, 3, stride=2)
    arr = np.random.randn(1, 20, 312)
    input = flow.Tensor(arr)
    output = model(input)
    print(output.shape)
