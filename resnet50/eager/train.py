import argparse
import numpy as np
import os
import time
import datetime
import shutil
from tqdm import tqdm
import oneflow as flow

import sys
sys.path.append(".")
# from models.resnet50 import resnet50
# from utils.ofrecord_data_utils import OFRecordDataLoader
from graph.train_consistent import prepare_modules, parse_args

class AverageMeter:
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def train_one_epoch(args, model, criterion, data_loader, optimizer, epoch, lr_scheduler):
    model.train()
    optimizer.zero_grad()
    
    num_steps = len(data_loader)
    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    norm_meter = AverageMeter()

    start = time.time()
    end = time.time()
    for steps in range(len(data_loader)):
        # data setup
        image, label = data_loader()
        image = image.to("cuda")
        label = label.to("cuda")

        # calculate
        logits = model(image)
        loss = criterion(logits, label)

        # update model
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # update metrics
        loss_meter.update(loss.numpy(), args.train_batch_size_per_device)
        batch_time.update(time.time() - end)
        end = time.time()

        # print info
        if steps % args.loss_print_every_n_iter == 0:
            lr = optimizer.param_groups[0]['lr']
            print("Train:[%d/%d][%d/%d] Time: %.4f(%.4f) Training Loss: %.4f (%.4f) Lr: %.6f" % ((epoch + 1), args.num_epochs, steps, num_steps, batch_time.val, batch_time.avg, loss_meter.val, loss_meter.avg, lr))
        if steps >= 100:
            break

        # lr_scheduler.step()
    return loss_meter.avg

def valid(args, model, criterion, data_loader):
    # Validation
    eval_losses = AverageMeter()

    model.eval()
    all_preds, all_label = [], []

    for steps in tqdm(range(len(data_loader))):
        # get data
        image, label = data_loader()
        image = image.to("cuda")
        label = label.to("cuda")
        with flow.no_grad():
            logits = model(image)
            preds = logits.softmax()
            eval_loss = criterion(logits, label)
            eval_losses.update(eval_loss.numpy())

        preds = preds.numpy()
        preds = np.argmax(preds, axis=-1)
        label = label.numpy()

        # collect results
        if len(all_preds) == 0:
            all_preds.append(preds)
            all_label.append(label)
        else:
            all_preds[0] = np.append(
                all_preds[0], preds, axis=0
            )
            all_label[0] = np.append(
                all_label[0], label, axis=0
            )
    
    all_preds, all_label = all_preds[0], all_label[0]
    accuracy = simple_accuracy(all_preds, all_label)

    print("Validation Results")
    print("Valid Loss: %2.5f" % eval_losses.avg)
    print("Valid Accuracy: %2.5f" % accuracy)
    return accuracy       

def simple_accuracy(preds, labels):
    return (preds == labels).mean()

def save_checkpoint(model, save_path):
    flow.save(model.state_dict(), save_path)

def save_logs(training_info, file_path):
    writer = open(file_path, "w")
    for info in training_info:
        writer.write("%f\n" % info)
    writer.close()


def main(args):
    train_data_loader, val_data_loader, model, optimizer, criterion, lr_scheduler = prepare_modules(args, to_consistent=False)

    loss_list = []
    accuracy_list = []
    best_acc = 0.0
    for epoch in range(args.num_epochs):
        print("***** Runing Training *****")
        train_loss = train_one_epoch(args, model, criterion, train_data_loader, optimizer, epoch, lr_scheduler)
        break
        print("***** Run Validation *****")
        accuracy = valid(args, model, criterion, val_data_loader)
        
        # save model after each epoch
        print("***** Save Checkpoint *****")
        save_path = os.path.join(args.save_checkpoint_path, "epoch_%d_val_acc_%f" % (epoch, accuracy))
        save_checkpoint(model, save_path)
        print("Save checkpoint to: ", save_path)
        
        # save best model
        if best_acc < accuracy:
            save_path = os.path.join(args.save_checkpoint_path, "best_model")
            if os.path.exists(save_path):
                shutil.rmtree(save_path, True)
            save_checkpoint(model, save_path)
            best_acc = accuracy
        
        loss_list.append(train_loss)
        accuracy_list.append(accuracy)
    print("End Training!")
    print("Max Accuracy: ", best_acc)

    # saving training information
    print("***** Save Logs *****")
    save_logs(loss_list, "eager/losses.txt")
    print("Save loss info to: ", "eager/losses.txt")
    save_logs(accuracy_list, "eager/accuracy.txt")
    print("Save acc info to: ", "eager/accuracy.txt")

if __name__ == "__main__":
    args = parse_args()
    main(args)