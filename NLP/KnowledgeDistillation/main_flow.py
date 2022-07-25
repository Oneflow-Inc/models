import math
import argparse
import oneflow as flow
import oneflow.nn as nn
import oneflow.nn.functional as F
from flowvision import datasets, transforms
import oneflow.utils.data
from model import TeacherNet, StudentNet
import matplotlib.pyplot as plt

flow.manual_seed(0)
# flow.cuda.manual_seed(0)

def train_teacher(model, device, train_loader, optimizer, epoch):
    print('train_teacher')
    model.train()
    trained_samples = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()

        trained_samples += len(data)
        progress = math.ceil(batch_idx / len(train_loader) * 50)
        print(
            "\rTrain epoch %d: %d/%d, [%-51s] %d%%"
            % (
                epoch,
                trained_samples,
                len(train_loader.dataset),
                "-" * progress + ">",
                progress * 2,
            ),
            end="",
        )


def test_teacher(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with flow.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.cross_entropy(
                output, target, reduction="sum"
            ).item()  # sum up batch loss
            pred = output.argmax(
                dim=1, keepdim=True
            )  # get the index of the max log-probability
            # print('pred.shape', pred.shape)
            # print('target.shape', target.shape)
            correct += pred.eq(target.view(pred.shape)).sum().item()

    test_loss /= len(test_loader.dataset)

    print(
        "\nTest: average loss: {:.4f}, accuracy: {}/{} ({:.0f}%)".format(
            test_loss,
            correct,
            len(test_loader.dataset),
            100.0 * correct / len(test_loader.dataset),
        )
    )
    return test_loss, correct / len(test_loader.dataset)


def teacher_main():
    epochs = 10
    batch_size = 64
    flow.manual_seed(0)

    device = flow.device("cuda" if flow.cuda.is_available() else "cpu")

    train_loader = flow.utils.data.DataLoader(
        datasets.MNIST(
            './',
            train=True,
            download=True,
            transform=transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
            ),
        ),
        batch_size=batch_size,
        shuffle=True,
    )
    test_loader = flow.utils.data.DataLoader(
        datasets.MNIST(
            "./",
            train=False,
            download=True,
            transform=transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
            ),
        ),
        batch_size=1000,
        shuffle=True,
    )

    model = TeacherNet().to(device)
    optimizer = flow.optim.Adam(model.parameters())

    teacher_history = []

    for epoch in range(1, epochs + 1):
        train_teacher(model, device, train_loader, optimizer, epoch)
        loss, acc = test_teacher(model, device, test_loader)

        teacher_history.append((loss, acc))

    flow.save(model.state_dict(), "teacher")
    return model, teacher_history

if __name__ == '__main__':
    teacher_model, teacher_history = teacher_main()