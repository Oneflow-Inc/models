import torch
from torch import nn
#from oneflow.python.framework.function_util import global_function_or_identity
from models.rnn_model_pytorch import RNN_PYTORCH

from utils.dataset import *
import random
import time
import math
import numpy as np


# Find letter index from all_letters, e.g. "a" = 0
def letterToIndex(letter):
    return all_letters.find(letter)

# Just for demonstration, turn a letter into a <1 x n_letters> Tensor
def letterToTensor(letter):
    tensor = torch.zeros(1, n_letters)
    tensor[0][letterToIndex(letter)] = 1
    return tensor

# Turn a line into a <line_length x 1 x n_letters>,
# or an array of one-hot letter vectors
def lineToTensor(line):
    tensor = torch.zeros(len(line), 1, n_letters)
    for li, letter in enumerate(line):
        tensor[li][0][letterToIndex(letter)] = 1
    return tensor

def train(category_tensor, line_tensor):
    hidden = rnn.initHidden()

    rnn.zero_grad()

    for i in range(line_tensor.size()[0]):
        output, hidden = rnn(line_tensor[i], hidden)

    loss = criterion(output, category_tensor)
    loss.backward()

    # Add parameters' gradients to their values, multiplied by learning rate
    for p in rnn.parameters():
        p.data.add_(p.grad.data, alpha=-learning_rate)

    return output, loss.item()

def randomChoice(l):
    x = random.randint(0, len(l) - 1)
    return l[random.randint(0, len(l) - 1)]

def randomTrainingExample():
    category = randomChoice(all_categories)
    line = randomChoice(category_lines[category])
    category_tensor = torch.tensor([all_categories.index(category)], dtype=torch.long)
    line_tensor = lineToTensor(line)
    return category, line, category_tensor.to("cuda"), line_tensor.to("cuda")

def categoryFromOutput(output):
    top_n, top_i = output.topk(1)
    category_i = top_i[0].item()
    return all_categories[category_i], category_i

def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

learning_rate = 0.005 # If you set this too high, it might explode. If too low, it might not learn
dataset_path = "./data/names"
n_categories = processDataset(dataset_path)

n_hidden = 128
rnn = RNN_PYTORCH(n_letters, n_hidden, n_categories)
rnn.load_state_dict(torch.load("models/rnn"))
rnn.to("cuda")
criterion = nn.NLLLoss() #have n_categories as arguments here in pytorch tutorial
criterion.to("cuda")

current_loss = 0
all_losses = []

n_iters = 100000
print_every = 500
plot_every = 1000
start = time.time()
random.seed(10) #make sure the random sampling process is the same as pytorch version.

for iter in range(1, n_iters + 1):
    category, line, category_tensor, line_tensor = randomTrainingExample()
    output, loss = train(category_tensor, line_tensor)
    current_loss += loss

    # Print iter number, loss, name and guess
    if iter % print_every == 0:
        guess, guess_i = categoryFromOutput(output)
        correct = '✓' if guess == category else '✗ (%s)' % category
        print('%d %d%% (%s) %.4f %s / %s %s' % (iter, iter / n_iters * 100, timeSince(start), loss, line, guess, correct))

    # Add current loss avg to list of losses
    if iter % plot_every == 0:
        all_losses.append(current_loss / plot_every)
        current_loss = 0

writer = open("all_losses_pytorch.txt", "w")
for o in all_losses:
    writer.write("%f\n" % o)
writer.close()