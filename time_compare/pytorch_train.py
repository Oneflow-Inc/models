import torch
import numpy as np
from models.alexnet_pytorch import alexnet
import time
import random

def train_pytorch(args, image_nd, label_nd):
    learning_rate = args.learning_rate
    mom = args.mom
    warmup_iters = args.warmup_iters
    bp_iters = args.bp_iters

    for_time = 0.0
    bp_time = 0.0
    update_time = 0.0

    # model, data, optimizer, criterion setup
    model = alexnet().to("cuda")
    image = torch.tensor(image_nd).to("cuda")
    label = torch.tensor(label_nd, dtype=torch.long).to("cuda")
    optimizer = torch.optim.SGD(
        model.parameters(), lr=learning_rate, momentum=mom
    )
    criterion = torch.nn.CrossEntropyLoss()

    model.train()
    print("start pytorch training...")
    for i in range(warmup_iters + bp_iters):
        if i == warmup_iters:
            start_t = time.time()
            for_time = 0.0
            bp_time = 0.0
            update_time = 0.0
        
        torch.cuda.synchronize()
        s_t = time.time()
        logits = model(image)
        loss = criterion(logits, label)
        torch.cuda.synchronize()
        for_time += time.time() - s_t

        torch.cuda.synchronize()
        s_t = time.time()
        loss.backward()
        torch.cuda.synchronize()
        bp_time += time.time() - s_t

        torch.cuda.synchronize()
        s_t = time.time()
        optimizer.step()
        optimizer.zero_grad()
        torch.cuda.synchronize()
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
    