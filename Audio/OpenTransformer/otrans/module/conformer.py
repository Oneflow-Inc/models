import oneflow as flow
import oneflow.nn as nn
import oneflow.nn.functional as F


class ConformerConvolutionModule(nn.Module):
    def __init__(self, channels, kernel_size, bias=True, dropout=0.0):
        super(ConformerConvolutionModule, self).__init__()

        assert kernel_size % 2 == 1

        self.pointwise_conv1 = nn.Linear(channels, 2 * channels, bias=bias)

        self.depthwise_conv = nn.Conv1d(
            channels,
            channels,
            kernel_size,
            stride=1,
            padding=(kernel_size - 1) // 2,
            groups=channels,
            bias=bias,
        )

        self.batch_norm = nn.BatchNorm1d(channels)

        self.pointwise_conv2 = nn.Linear(channels, channels, bias=bias)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        """
        Args:
            x: [batch_size, time, channels]
            mask: [batch_size, time]
        """
        mask = mask.unsqueeze(2).repeat([1, 1, x.size(-1)])

        x = self.pointwise_conv1(x)
        x = F.glu(x)
        x = flow.masked_fill(x, mask == 0, 0.0)

        x = x.transpose(1, 2)
        x = self.depthwise_conv(x)
        x = self.batch_norm(x)
        x = x * flow.sigmoid(x)
        x = x.transpose(1, 2)

        x = self.pointwise_conv2(x)
        x = flow.masked_fill(x, mask == 0, 0.0)

        return x
