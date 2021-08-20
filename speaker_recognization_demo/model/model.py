import oneflow.nn as nn


class simple_CNN(nn.Module):
    def __init__(self) -> None:
        super(simple_CNN, self).__init__()
        self.convs = nn.Sequential(
            nn.Conv1d(1, 16, 100, stride=10),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Conv1d(16, 64, 21, stride=10),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 64, 5, stride=5),
            nn.BatchNorm1d(64),
            nn.ReLU(),
        )
        self.linears = nn.Sequential(nn.Linear(1 * 6 * 64, 128), nn.Linear(128, 2))

    def forward(self, x):
        x = self.convs(x)
        x = x.view(x.size(0), -1)
        x = self.linears(x)

        return x
