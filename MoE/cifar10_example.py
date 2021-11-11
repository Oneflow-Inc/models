# This file is based on the `https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html`.

import oneflow as torch
import oneflow as flow
import oneflow.nn as nn
import oneflow.nn.functional as F
from moe import MoE
import oneflow.optim as optim

import flowvision as vision

transform = vision.transforms.Compose(
    [vision.transforms.ToTensor(),
     vision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = vision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = flow.utils.data.DataLoader(trainset, batch_size=64,
                                          shuffle=True, num_workers=1)

testset = vision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = flow.utils.data.DataLoader(testset, batch_size=64,
                                         shuffle=False, num_workers=1)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


device = flow.device('cpu')
net = MoE(input_size=3072,output_size= 10, num_experts=10, hidden_size=256, noisy_gating=True, k=4)
net.to(device)


optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
criterion = nn.CrossEntropyLoss()
criterion.to(device)

for epoch in range(10):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        inputs = inputs.view(inputs.shape[0], -1)
        outputs, aux_loss = net(inputs)
        loss = criterion(outputs, labels)
        total_loss = loss + aux_loss
        total_loss.backward()
        optimizer.step()


        # print statistics
        running_loss += loss.item()
        if i % 100 == 99:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 100))
            running_loss = 0.0

print('Finished Training')


correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs, _ = net(images.view(images.shape[0], -1))
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the 10000 test images: %d %%' % (
    100 * correct / total))

# yields a test accuracy of around 39 %
