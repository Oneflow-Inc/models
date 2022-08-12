import math
import argparse
import matplotlib.pyplot as plt
import os

import oneflow as flow
import oneflow.nn as nn
import oneflow.nn.functional as F
from flowvision import datasets, transforms

from model import TeacherNet, StudentNet


def train_teacher(model, train_loader, optimizer, epoch):
    model.train()
    trained_samples = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(args.device), target.to(args.device)
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


def test_teacher(model, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with flow.no_grad():
        for data, target in test_loader:
            data, target = data.to(args.device), target.to(args.device)
            output = model(data)
            test_loss += F.cross_entropy(
                output, target, reduction="sum"
            ).item()  # sum up batch loss
            pred = output.argmax(
                dim=1, keepdim=True
            )  # get the index of the max log-probability
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
    train_loader = flow.utils.data.DataLoader(
        datasets.MNIST(
            "./",
            train=True,
            download=True,
            transform=transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
            ),
        ),
        batch_size=args.batch_size,
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

    model = TeacherNet().to(args.device)
    optimizer = flow.optim.Adadelta(model.parameters())

    teacher_history = []

    for epoch in range(1, args.epochs + 1):
        train_teacher(model, train_loader, optimizer, epoch)
        loss, acc = test_teacher(model, test_loader)

        teacher_history.append((loss, acc))

    flow.save(
        model.state_dict(), args.model_save_dir + "/teacher"
    )  # If the path does not exist, oneflow will automatically create it.
    return model, teacher_history


def distillation(y, labels, teacher_scores, temp, alpha):
    return nn.KLDivLoss()(
        F.log_softmax(y / temp, dim=1), F.softmax(teacher_scores / temp, dim=1)
    ) * (temp * temp * 2.0 * alpha) + F.cross_entropy(y, labels) * (1.0 - alpha)


def train_student_kd(model, train_loader, optimizer, epoch):
    model.train()
    trained_samples = 0
    # load teacher model
    teacher_model = TeacherNet().to(args.device)
    teacher_model.load_state_dict(flow.load(args.load_teacher_checkpoint_dir))
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(args.device), target.to(args.device)
        optimizer.zero_grad()
        output = model(data)
        teacher_output = teacher_model(data)
        teacher_output = (
            teacher_output.detach()
        )  # Cut off the back propagation of teacher network
        loss = distillation(
            output, target, teacher_output, temp=args.temperature, alpha=args.alpha
        )
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


def test_student_kd(model, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with flow.no_grad():
        for data, target in test_loader:
            data, target = data.to(args.device), target.to(args.device)
            output = model(data)
            test_loss += F.cross_entropy(
                output, target, reduction="sum"
            ).item()  # sum up batch loss
            pred = output.argmax(
                dim=1, keepdim=True
            )  # get the index of the max log-probability
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


def student_kd_main():

    train_loader = flow.utils.data.DataLoader(
        datasets.MNIST(
            "./",
            train=True,
            download=True,
            transform=transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
            ),
        ),
        batch_size=args.batch_size,
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

    model = StudentNet().to(args.device)
    optimizer = flow.optim.Adadelta(model.parameters())

    student_history = []
    for epoch in range(1, args.epochs + 1):
        train_student_kd(model, train_loader, optimizer, epoch)
        loss, acc = test_student_kd(model, test_loader)
        student_history.append((loss, acc))

    flow.save(
        model.state_dict(), args.model_save_dir + "/student_kd"
    )  # If the path does not exist, oneflow will automatically create it.
    return model, student_history


def train_student(model, train_loader, optimizer, epoch):
    model.train()
    trained_samples = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(args.device), target.to(args.device)
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


def test_student(model, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with flow.no_grad():
        for data, target in test_loader:
            data, target = data.to(args.device), target.to(args.device)
            output = model(data)
            test_loss += F.cross_entropy(
                output, target, reduction="sum"
            ).item()  # sum up batch loss
            pred = output.argmax(
                dim=1, keepdim=True
            )  # get the index of the max log-probability
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


def student_main():

    train_loader = flow.utils.data.DataLoader(
        datasets.MNIST(
            "./",
            train=True,
            download=True,
            transform=transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
            ),
        ),
        batch_size=args.batch_size,
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

    model = StudentNet().to(args.device)
    optimizer = flow.optim.Adadelta(model.parameters())

    student_history = []

    for epoch in range(1, args.epochs + 1):
        train_student(model, train_loader, optimizer, epoch)
        loss, acc = test_student(model, test_loader)
        student_history.append((loss, acc))

    flow.save(
        model.state_dict(), args.model_save_dir + "/student"
    )  # If the path does not exist, oneflow will automatically create it.
    return model, student_history


if __name__ == "__main__":
    flow.manual_seed(2022)
    flow.cuda.manual_seed(2022)
    flow.cuda.manual_seed_all(2022)

    parser = argparse.ArgumentParser(description="flags for knowledge distillation")
    # Training Parameters
    parser.add_argument(
        "--model_save_dir",
        type=str,
        default="./output/model_save",
        required=False,
        help="model save directory",
    )
    parser.add_argument(
        "--image_save_name",
        type=str,
        default="./output/images/result.jpg",
        required=False,
        help="image save name",
    )
    parser.add_argument(
        "--load_teacher_checkpoint_dir", type=str, default="./output/model_save/teacher"
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default="teacher",
        choices=["teacher", "student_kd", "student", "compare"],
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=128)
    # Model Parameters
    parser.add_argument("--temperature", type=float, default=5.0)
    parser.add_argument("--alpha", type=float, default=0.7)

    args = parser.parse_args()
    args.device = flow.device("cuda" if flow.cuda.is_available() else "cpu")

    teacher_history = None
    student_kd_history = None
    student_simple_history = None

    if args.model_type == "teacher" or args.model_type == "compare":
        print("===== start training teacher model ... =====")
        teacher_model, teacher_history = teacher_main()
    if args.model_type == "student_kd" or args.model_type == "compare":
        print("===== start training student_kd model ... =====")
        student_kd_model, student_kd_history = student_kd_main()
    if args.model_type == "student" or args.model_type == "compare":
        print("===== start training student model ... =====")
        student_simple_model, student_simple_history = student_main()

    x = list(range(1, args.epochs + 1))

    plt.subplot(2, 1, 1)
    if teacher_history != None:
        plt.plot(
            x, [teacher_history[i][1] for i in range(args.epochs)], label="teacher"
        )
    if student_kd_history != None:
        plt.plot(
            x,
            [student_kd_history[i][1] for i in range(args.epochs)],
            label="student with KD",
        )
    if student_simple_history != None:
        plt.plot(
            x,
            [student_simple_history[i][1] for i in range(args.epochs)],
            label="student without KD",
        )

    plt.title("Test accuracy")
    plt.legend()

    plt.subplot(2, 1, 2)
    if teacher_history != None:
        plt.plot(
            x, [teacher_history[i][0] for i in range(args.epochs)], label="teacher"
        )
    if student_kd_history != None:
        plt.plot(
            x,
            [student_kd_history[i][0] for i in range(args.epochs)],
            label="student with KD",
        )
    if student_simple_history != None:
        plt.plot(
            x,
            [student_simple_history[i][0] for i in range(args.epochs)],
            label="student without KD",
        )

    plt.title("Test loss")
    plt.legend()

    directory = os.path.abspath(
        os.path.dirname(args.image_save_name) + os.path.sep + "."
    )  # If the path does not exist, create it.
    if not os.path.exists(directory):
        os.makedirs(directory)
    plt.savefig(args.image_save_name)
