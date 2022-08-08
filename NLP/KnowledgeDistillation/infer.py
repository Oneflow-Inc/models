import argparse
import matplotlib.pyplot as plt
import numpy as np
import time

import oneflow as flow
from flowvision import datasets, transforms

from model import TeacherNet, StudentNet


def softmax_t(x, t):
    x_exp = np.exp(x / t)
    return x_exp / np.sum(x_exp)

if __name__ == '__main__':

    parser = argparse.ArgumentParser("flags for test resnet50")
    parser.add_argument(
        "--model_load_dir", type=str, default="./output/model_save/teacher"
    )
    parser.add_argument('--model_type', type=str, default="teacher", choices=["teacher", "student_kd", "student"])
    parser.add_argument('--picture_index', type=int, default=0)
    parser.add_argument('--temperature', type=float, default=10.0)
    parser.add_argument("--image_save_name", type=str, default="./output/images/infer.jpg",
        required=False, help="images save name")

    args = parser.parse_args()
    args.device = flow.device("cuda" if flow.cuda.is_available() else "cpu")

    start_t = time.perf_counter()
    print("***** Model Init *****")
    if args.model_type == 'teacher':
        model = TeacherNet()
    else: # student_ks, student
        model = StudentNet()
    model.load_state_dict(flow.load(args.model_load_dir))
    end_t = time.perf_counter()
    print(f"***** Model Init Finish, time escapled {end_t - start_t:.6f} s *****")

    model = model.to(args.device)
    model.eval()

    # dataset
    dataset = datasets.MNIST(
            './',
            train=False,
            download=True,
            transform=transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
            ),
        )
    subset_indices = [args.picture_index]
    subset = flow.utils.data.Subset(dataset, subset_indices)
    # dataloader
    data_loader = flow.utils.data.DataLoader(
        subset,
        batch_size=1,
        shuffle=False,
    )
    with flow.no_grad():
        data, target = next(iter(data_loader))
        data, target = data.to(args.device), target.to(args.device)
        output = model(data)

    test_x = data.cpu().numpy()
    y_out = output.cpu().numpy()
    y_out = y_out[0, ::]
    print('Output (NO softmax):', y_out)
    print("the number is", flow.argmax(output).cpu().numpy())

    plt.subplot(3, 1, 1)
    plt.imshow(test_x[0, 0, ::])

    plt.subplot(3, 1, 2)
    plt.bar(list(range(10)), softmax_t(y_out, 1), width=0.3)

    plt.subplot(3, 1, 3)
    plt.bar(list(range(10)), softmax_t(y_out, args.temperature), width=0.3)
    plt.savefig(args.image_save_name)
    print("picture saved.")
