import tqdm

from typing import Optional

import torch
import torch.nn as nn
from optimization_pt import get_scheduler

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
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
    
    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class Trainer:
    """ Trainer for GPT model. """

    def __init__(
        self,
        model,
        train_dataloader=None,
        test_dataloader=None,
        epoch: int = 1,
        lr: float = 1e-4,
        betas=(0.9, 0.999),
        weight_decay: float = 0.01,
        warmup_steps: Optional[int] = None,
        accumulate_gradient_steps: int = 1,
        output_path=None,
    ):
        """
        :param model: model
        :param train_dataloader: train dataset data loader
        :param test_dataloader: test dataset data loader [can be None]
        :param lr: learning rate of optimizer
        :param betas: Adam optimizer betas
        :param weight_decay: Adam optimizer weight decay param
        """

        self.model = model.cuda()
        self.output_path = output_path

        # Setting the train and test data loader
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        self.epoch = epoch
        self.accumulate_gradient_steps = accumulate_gradient_steps

        # # Setting the Adam optimizer with hyper-param
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, betas=betas, weight_decay=weight_decay)
        
        total_train_steps = len(self.train_dataloader) * self.epoch
        self.lr_scheduler = get_scheduler('linear', self.optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_train_steps)
        print("Total Parameters:", sum([p.nelement() for p in self.model.parameters()]))

    def train(self):
        for i in range(self.epoch):
            self.train_single_epoch(self.train_dataloader, i)
            self.evaluate(self.test_dataloader, i)
            self.save(i + 1, file_path="checkpoints/")
        torch.save(self.model.state_dict(), output_path)

    def test(self):
        self.evaluate(self.test_dataloader)

    def train_single_epoch(self, data_loader, epoch):
        self.model.train()

        losses = AverageMeter("loss")

        self.optimizer.zero_grad()
        data_iter = tqdm.tqdm(data_loader, desc="Training: %0d" % (epoch), total=len(data_loader))
        for step, batch in enumerate(data_iter):
            inputs, labels = (batch, batch)
            inputs = inputs.cuda()
            labels = labels.cuda()
            outputs = self.model(inputs, labels=labels)
            loss = outputs[0]

            losses.update(loss.item())

            if step % self.accumulate_gradient_steps == 0:
                loss.backward()
                self.lr_scheduler.step()
                self.optimizer.step()
                self.optimizer.zero_grad()

            logging = {
                'epoch': epoch,
                'step': step,
                'avg_loss': losses.avg,
                'loss': losses.val,
                'lr': self.lr_scheduler.get_lr()[0]
            }
            data_iter.set_postfix(logging)

        print("Training:%0d, avg_loss:%.4f" % (epoch, losses.avg))

    def evaluate(self, data_loader, epoch=0):
        self.model.eval()
        losses = AverageMeter("loss")

        data_iter = tqdm.tqdm(data_loader, desc="Evaluate: ", total=len(data_loader))
        for step, batch in enumerate(data_iter):
            with torch.no_grad():
                inputs, labels = (batch, batch)
                inputs = inputs.cuda()
                labels = labels.cuda()
                outputs = self.model(inputs, labels=labels)
            loss = outputs[0]
            
            loss_item = loss.item()
            losses.update(loss_item)
            
            logging = {
                'epoch': epoch,
                'step': step,
                'avg_loss': losses.avg,
                'loss': losses.val,
            }
            data_iter.set_postfix(logging)

        print("Evaluating:%0d, avg_loss:%.4f" % (epoch, losses.avg))


    def save(self, epoch, file_path="checkpoints/"):
        """
        Saving the current model on file_path

        :param epoch: current epoch number
        :param file_path: model output path which gonna be file_path+"ep%d" % epoch
        :return: final_output_path
        """
        output_path = file_path + "epoch%d.pt" % epoch
        torch.save(self.model.state_dict(), output_path)
        print("EP:%d Model Saved on:" % epoch, output_path)
        return output_path
