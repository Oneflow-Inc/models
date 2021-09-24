"""
@author: Chenhao Lu <luchenhao@zhejianglab.com>
@author: Yizhang Wang <1739601638@qq.com>
"""
import oneflow as flow


class DeepQNetwork(flow.nn.Module):
    def __init__(self):
        super(DeepQNetwork, self).__init__()

        self.conv1 = flow.nn.Sequential(
            flow.nn.Conv2d(4, 32, kernel_size=8, stride=4), flow.nn.ReLU(inplace=True)
        )
        self.conv2 = flow.nn.Sequential(
            flow.nn.Conv2d(32, 64, kernel_size=4, stride=2), flow.nn.ReLU(inplace=True)
        )
        self.conv3 = flow.nn.Sequential(
            flow.nn.Conv2d(64, 64, kernel_size=3, stride=1), flow.nn.ReLU(inplace=True)
        )

        self.fc1 = flow.nn.Sequential(
            flow.nn.Linear(7 * 7 * 64, 512), flow.nn.ReLU(inplace=True)
        )

        self.fc2 = flow.nn.Linear(512, 2)
        self._create_weights()

    def _create_weights(self):
        for m in self.modules():
            if isinstance(m, flow.nn.Conv2d) or isinstance(m, flow.nn.Linear):
                flow.nn.init.uniform_(m.weight, -0.01, 0.01)
                flow.nn.init.constant_(m.bias, 0)

    def forward(self, input):
        assert input.device.type == "cuda"
        output = self.conv1(input)
        output = self.conv2(output)
        output = self.conv3(output)
        output = flow.reshape(output, shape=(output.size(0), -1))
        output = self.fc1(output)
        output = self.fc2(output)

        return output
