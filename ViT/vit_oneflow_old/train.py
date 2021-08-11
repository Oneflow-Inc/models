import argparse
import numpy as np
import os
import time
import logging

from utils.ofrecord_data_utils import OFRecordDataLoader
from utils.lr_scheduler import WarmupCosineSchedule, WarmupLinearSchedule
from tqdm import tqdm
from models import CONFIGS, VisionTransformer


import oneflow
import oneflow.F as F
import oneflow as flow
from oneflow import nn
from oneflow import optim
import random

logger = logging.getLogger(__name__)

class AverageMeter(object):
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

def simple_accuracy(preds, labels):
    return (preds == labels).mean()

def build_loader(args):
    train_data_loader = OFRecordDataLoader(
        ofrecord_root=args.data_path,
        mode="train",
        dataset_size=args.num_train_examples,
        batch_size=args.train_batch_size,
    )

    val_data_loader = OFRecordDataLoader(
        ofrecord_root=args.data_path,
        mode="validation",
        dataset_size=args.num_val_examples,
        batch_size=args.eval_batch_size,
    )
    return train_data_loader, val_data_loader


def setup(args):
    # Prepare model
    config = CONFIGS[args.model_type]
    if args.dataset == "imagenet":
        num_classes = 1000

    model = VisionTransformer(config, args.img_size, zero_head=True, num_classes=num_classes)
    if args.pretrained_dir:
        model.load_state_dict(flow.load(args.pretrained_dir), strict=False)
    model.to("cuda")


    logger.info("{}".format(config))
    return args, model

# def clip_gradient(optimizer, grad_clip):
#     """
#     Clips gradients computed during backpropagation to avoid explosion of gradients.

#     :param optimizer: optimizer with the gradients to be clipped
#     :param grad_clip: clip value
#     """
#     for group in optimizer.param_groups:
#         for param in group["params"]:
#             if param.grad is not None:
#                 param.grad.data.clamp_(-grad_clip, grad_clip)

def train(args, model):
    # add gradient steps
    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps
    
    # Prepare dataset
    train_data_loader, test_data_loader = build_loader(args)

    # Prepare optimizer and scheduler
    optimizer = optim.SGD(model.parameters(),
                          lr = args.learning_rate,
                          momentum = 0.9)
    
    # define criterion
    criterion = flow.nn.CrossEntropyLoss().to("cuda")

    # total fine-tune steps
    t_total = args.num_steps
    if args.decay_type == "cosine":
        scheduler = WarmupCosineSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=t_total)
    else:
        scheduler = WarmupLinearSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=t_total)

    # Train
    # logger.info("***** Running training *****")
    # logger.info("  Total optimization steps = %d", args.num_steps)
    # logger.info("  Instantaneous batch size per GPU = %d", args.train_batch_size)
    # logger.info("  Total train batch size (w. parallel & accumulation) = %d",
    #             args.train_batch_size * args.gradient_accumulation_steps)
    # logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)


    print("***** Runing Training *****")
    print("  Total optimization steps = %d" % args.num_steps)
    print("  Instantaneous batch size per GPU = %d" % args.train_batch_size)
    print("  Total train batch size (w. parallel & accumulation) = %d" %
                (args.train_batch_size * args.gradient_accumulation_steps))
    print("  Gradient Accumulation steps = %d" % args.gradient_accumulation_steps)

    # model.zero_grad()
    losses = AverageMeter()
    batch_time = AverageMeter()
    global_step, best_acc = 0, 0
    while True:
        model.train()
        for step in range(len(train_data_loader)):
            image, label = train_data_loader.get_batch()

            # oneflow train
            image = image.to("cuda")
            label = label.to("cuda")
            logits = model(image)
            loss = criterion(logits, label)
            
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps
            
            loss.backward()

            if (step + 1) % args.gradient_accumulation_steps == 0:
                losses.update(loss.numpy() * args.gradient_accumulation_steps)
                # clip_gradient(optimizer, 1.0)
                scheduler.step()

                optimizer.step()
                optimizer.zero_grad()
                global_step += 1
                
                # current learning rate
                learning_rate = optimizer.param_groups[0]['lr']

                print(
                    "Training (%d / %d Steps) (loss=%2.5f) (learning_rate=%2.5f)" % (global_step, t_total, losses.val, learning_rate)
                )

                if global_step % args.eval_every == 0:
                    accuracy = valid(args, model, test_data_loader, global_step)
                    if best_acc < accuracy:
                        best_acc = accuracy
                    model.train()
                
                if global_step % t_total == 0:
                    break
    
        losses.reset()
        if global_step % t_total == 0:
            break
    

    print("Best Accuracy: \t%f" % best_acc)
    print("End Training!")


def valid(args, model, test_data_loader, global_step):
    # Validation
    eval_losses = AverageMeter()
    print("***** Run Validation *****")
    print(" Num Steps = %d" % len(test_data_loader))
    print("  Batch size = %d" % args.eval_batch_size)
    
    model.eval()
    all_preds, all_label = [], []

    loss_func = nn.CrossEntropyLoss().to("cuda")
    for steps in tqdm(range(len(test_data_loader))):
        image, label = test_data_loader.get_batch()
        image = image.to("cuda")
        label = label.to("cuda")
        with flow.no_grad():
            logits = model(image)
            eval_loss = loss_func(logits, label)
            eval_losses.update(eval_loss.numpy())

        # turn tensor to numpy.ndarray
        preds = logits.numpy()
        preds = np.argmax(preds, axis=-1)
        label_numpy = label.numpy()

        # collect results
        if len(all_preds) == 0:
            all_preds.append(preds)
            all_label.append(label_numpy)
        else:
            all_preds[0] = np.append(
                all_preds[0], preds, axis=0
            )
            all_label[0] = np.append(
                all_label[0], label_numpy, axis=0
            )

    
    all_preds, all_label = all_preds[0], all_label[0]
    accuracy = simple_accuracy(all_preds, all_label)

    # logger.info("\n")
    # logger.info("Validation Results")
    # logger.info("Global Steps: %d" % global_step)
    # logger.info("Valid Loss: %2.5f" % eval_losses.avg)
    # logger.info("Valid Accuracy: %2.5f" % accuracy)

    # use print instead of logger
    print()
    print("Validation Results")
    print("Global Step: %d" % global_step)
    print("Valid Loss: %2.5f" % eval_losses.avg)
    print("Valid Accuracy: %2.5f" % accuracy)
    #TODO
    # add tensorboard
    return accuracy
        



def main():
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument("--name", required=True,
                        help="Name of this run. Used for monitoring.")
    parser.add_argument("--dataset", choices=["cifar10", "cifar100", "imagenet"], default="imagenet",
                        help="Which downstream task.")
    parser.add_argument("--data_path", type=str, default="/data/imagenet/ofrecord/",
                        help="Which downstream task.")
    parser.add_argument("--model_type", choices=["ViT-B_16", "ViT-B_32", "ViT-L_16",
                                                 "ViT-L_32", "ViT-H_14", "R50-ViT-B_16"],
                        default="ViT-B_16",
                        help="Which variant to use.")
    parser.add_argument("--pretrained_dir", type=str, default="/data/rentianhe/weight/vit_oneflow/ViT-B_16_oneflow",
                        help="Where to search for pretrained ViT models.")
    parser.add_argument("--output_dir", default="output", type=str,
                        help="The output directory where checkpoints will be written.")
    parser.add_argument("--img_size", default=224, type=int,
                        help="Resolution size")
    parser.add_argument("--train_batch_size", default=512, type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size", default=32, type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--eval_every", default=100, type=int,
                        help="Run prediction on validation set every so many steps."
                             "Will always run one evaluation at the end of training.")
    parser.add_argument("--learning_rate", default=3e-2, type=float,
                        help="The initial learning rate for SGD.")
    parser.add_argument("--weight_decay", default=0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--num_steps", default=20000, type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--decay_type", choices=["cosine", "linear"], default="cosine",
                        help="How to decay the learning rate.")
    parser.add_argument("--warmup_steps", default=500, type=int,
                        help="Step of training to perform learning rate warmup for.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=32,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--num_train_examples", type=int, default=1281167, 
                        help="training picture number")
    parser.add_argument("--num_val_examples", type=int, default=50000, 
                        help="validation picture number")
    args = parser.parse_args()


    # model & criterion setup
    args, model= setup(args)

    # Training
    train(args, model)

if __name__ == "__main__":
    main()

