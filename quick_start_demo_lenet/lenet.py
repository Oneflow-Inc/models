# coding=utf-8
import numpy as np
import oneflow as flow
import oneflow.nn as nn

# define LeNet module
class LeNet5(nn.Module):
    def __init__(self, n_classes):
        super(LeNet5, self).__init__()
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1),
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2),
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1),
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2),
            nn.Conv2d(in_channels=16, out_channels=120, kernel_size=5, stride=1),
            nn.Tanh(),
        )
        self.classifier = nn.Sequential(
            nn.Linear(in_features=120, out_features=84),
            nn.Tanh(),
            nn.Linear(in_features=84, out_features=n_classes),
        )

    def forward(self, x):
        x = self.feature_extractor(x)
        x = flow.flatten(x, 1)
        logits = self.classifier(x)
        probs = flow.softmax(logits, dim=1)
        return logits, probs


# enable eager mode


# init model
model = LeNet5(10)
criterion = nn.CrossEntropyLoss()

# enable module to use cuda
model.to("cuda")
criterion.to("cuda")

learning_rate = 0.005
optimizer = flow.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

# generate random data and label
train_data = flow.Tensor(
    np.random.uniform(size=(30, 1, 32, 32)).astype(np.float32), device="cuda"
)
train_label = flow.Tensor(
    np.random.uniform(size=(30)).astype(np.int32), dtype=flow.int32, device="cuda"
)

# run forward, backward and update parameters
logits, probs = model(train_data)
loss = criterion(logits, train_label)
loss.backward()
optimizer.step()
optimizer.zero_grad()

print(loss.numpy())
