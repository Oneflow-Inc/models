import oneflow as flow
import numpy as np
from models.alexnet_oneflow import alexnet
import time
import random

def train_oneflow(args, image_nd, label_nd):
    learning_rate = args.learning_rate
    mom = args.mom
    warmup_iters = args.warmup_iters
    bp_iters = args.bp_iters

    for_time = 0.0
    bp_time = 0.0
    update_time = 0.0

    # model, data, optimizer, criterion setup
    model = alexnet().to("cuda")
    image = flow.tensor(image_nd).to("cuda")
    label = flow.tensor(label_nd).to("cuda")
    optimizer = flow.optim.SGD(
        model.parameters(), lr=learning_rate, momentum=mom
    )
    criterion = flow.nn.CrossEntropyLoss()

    model.train()
    print("start oneflow training...")
    for i in range(warmup_iters + bp_iters):
        if i == warmup_iters:
            start_t = time.time()
            for_time = 0.0
            bp_time = 0.0
            update_time = 0.0
        

        s_t = time.time()
        logits = model(image)
        loss = criterion(logits, label)
        for_time += time.time() - s_t


        s_t = time.time()
        loss.backward()
        bp_time += time.time() - s_t

        s_t = time.time()
        optimizer.step()
        optimizer.zero_grad()
        update_time += time.time() - s_t

        logits.cpu()
    end_t = time.time()

    print(
        "pytorch traning loop avg time : {}".format(
            ((end_t - start_t) ) / bp_iters
        )
    )
    print("forward avg time : {}".format((for_time ) / bp_iters))
    print("backward avg time : {}".format((bp_time ) / bp_iters))
    print("update parameters avg time : {}".format((update_time ) / bp_iters))
    