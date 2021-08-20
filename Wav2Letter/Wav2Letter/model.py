import oneflow.nn as nn


class Wav2Letter(nn.Module):

    def __init__(self, num_features, num_classes):
        super(Wav2Letter, self).__init__()

        self.layers = nn.Sequential(
            nn.Conv1d(num_features, 250, 48, 2),
            nn.ReLU(),
            nn.Conv1d(250, 250, 7),
            nn.ReLU(),
            nn.Conv1d(250, 250, 7),
            nn.ReLU(),
            nn.Conv1d(250, 250, 7),
            nn.ReLU(),
            nn.Conv1d(250, 250, 7),
            nn.ReLU(),
            nn.Conv1d(250, 250, 7),
            nn.ReLU(),
            nn.Conv1d(250, 250, 7),
            nn.ReLU(),
            nn.Conv1d(250, 250, 7),
            nn.ReLU(),
            nn.Conv1d(250, 2000, 32),
            nn.ReLU(),
            nn.Conv1d(2000, 2000, 1),
            nn.ReLU(),
            nn.Conv1d(2000, num_classes, 1),
        )

    def forward(self, data_batch):
        """Forward pass through Wav2Letter network than 
            takes log probability of output

        Args:
            data_batch (int): mini batch of data
             shape (batch, num_features, frame_len)

        Returns:
            log_probs (torch.Tensor):
                shape  (batch_size, num_classes, output_len)
        """
        y_pred = self.layers(data_batch)
        log_probs = nn.LogSoftmax(dim=1)(y_pred)

        return log_probs
