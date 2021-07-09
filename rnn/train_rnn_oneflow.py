import oneflow.experimental as flow
from oneflow.experimental import optim
import oneflow.experimental.nn as nn

from utils.dataset import *
from utils.tensor_utils import *
from models.rnn_model import RNN
import argparse

import time
import math
import numpy as np


def _parse_args():
    parser = argparse.ArgumentParser("flags for compare oneflow and pytorch speed")
    parser.add_argument(
        "--seed", nargs="?", type=int, const=0, help="specify random seed"
    )
    return parser.parse_args()


def train(category_tensor, line_tensor, rnn, criterion, of_sgd):
    hidden = rnn.initHidden()
    for i in range(line_tensor.size()[0]):
        output, hidden = rnn(line_tensor[i], hidden)
    loss = criterion(output, category_tensor)
    loss.backward()
    of_sgd.step()
    of_sgd.zero_grad()
    return output, loss.numpy()[0]


# refer to: https://blog.csdn.net/Nin7a/article/details/107631078
def topk_(matrix, K, axis=1):
    if axis == 0:
        row_index = np.arange(matrix.shape[1 - axis])
        topk_index = np.argpartition(-matrix, K, axis=axis)[0:K, :]
        topk_data = matrix[topk_index, row_index]
        topk_index_sort = np.argsort(-topk_data, axis=axis)
        topk_data_sort = topk_data[topk_index_sort, row_index]
        topk_index_sort = topk_index[0:K, :][topk_index_sort, row_index]
    else:
        column_index = np.arange(matrix.shape[1 - axis])[:, None]
        topk_index = np.argpartition(-matrix, K, axis=axis)[:, 0:K]
        topk_data = matrix[column_index, topk_index]
        topk_index_sort = np.argsort(-topk_data, axis=axis)
        topk_data_sort = topk_data[column_index, topk_index_sort]
        topk_index_sort = topk_index[:, 0:K][column_index, topk_index_sort]
    return topk_data_sort, topk_index_sort


def categoryFromOutput(output):
    top_n, top_i = topk_(output.numpy(), 1)
    category_i = top_i[0][0]
    return all_categories[category_i], category_i


def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return now, "%ds" % s


n_iters = 100000
print_every = 500
plot_every = 1000
learning_rate = (
    0.005  # If you set this too high, it might explode. If too low, it might not learn
)
# decrease learning rate if loss goes to NaN, increase learnig rate if it learns too slow


def main(args):
    flow.env.init()
    flow.enable_eager_execution()
    random.seed(args.seed)
    dataset_path = "./data/names"
    n_categories = processDataset(dataset_path)

    n_hidden = 128
    rnn = RNN(n_letters, n_hidden, n_categories)
    criterion = nn.NLLLoss()

    rnn.to("cuda")
    criterion.to("cuda")
    of_sgd = optim.SGD(rnn.parameters(), lr=learning_rate)

    # Keep track of losses for plotting
    current_loss = 0
    all_losses = []
    start = time.time()

    samples = 0.0
    correct_guess = 0.0
    for iter in range(1, n_iters + 1):
        category, line, category_tensor, line_tensor = randomTrainingExample()
        output, loss = train(category_tensor, line_tensor, rnn, criterion, of_sgd)
        current_loss += loss

        # Print iter number, loss, name and guess
        if iter % print_every == 0:
            start, time_str = timeSince(start)
            guess, guess_i = categoryFromOutput(output)
            correct = "✓" if guess == category else "✗ (%s)" % category
            if correct == "✓":
                correct_guess += 1
            samples += 1
            print(
                "iter: %d / %f%%, time_for_every_%d_iter: %s, loss: %.4f, predict: %s / %s, correct? %s, acc: %f"
                % (
                    iter,
                    float(iter) / n_iters * 100,
                    print_every,
                    time_str,
                    loss,
                    line,
                    guess,
                    correct,
                    correct_guess / samples,
                )
            )

        # Add current loss avg to list of losses
        if iter % plot_every == 0:
            all_losses.append(current_loss / plot_every)
            current_loss = 0

    writer = open("all_losses.txt", "w")
    for o in all_losses:
        writer.write("%f\n" % o)
    writer.close()


if __name__ == "__main__":
    args = _parse_args()
    main(args)
