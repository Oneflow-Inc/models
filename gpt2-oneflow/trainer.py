import tqdm

import oneflow as flow
import oneflow.nn as nn

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
        lr: float = 1e-4,
        betas=(0.9, 0.999),
        weight_decay: float = 0.01,
        warmup_steps=10000,
        device=None,
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

        self.device = flow.device("cpu") if device is None else device
        self.model = model
        self.model.to(self.device)

        # Setting the train and test data loader
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader

        # # Setting the Adam optimizer with hyper-param
        self.optimizer = flow.optim.Adam(self.model.parameters(), lr=lr, betas=betas, weight_decay=weight_decay)
        # self.lr_scheduler = flow.optim.lr_scheduler.CosineAnnealingLR(optimizer, steps=10000, alpha=0.0)
        print("Total Parameters:", sum([p.nelement() for p in self.model.parameters()]))

    def train(self, epoch):
        for i in range(epoch):
            self.train_single_epoch(self.train_dataloader, i)
            self.evaluate(self.test_dataloader, i)
            self.save(self, epoch, file_path="checkpoints")

    def test(self, epoch):
        self.evaluate(self.test_dataloader, epoch)

    def train_single_epoch(self, data_loader, epoch):
        self.model.train()

        losses = AverageMeter("loss")

        # self.optimizer.zero_grad()
        data_iter = tqdm.tqdm(data_loader, desc="Training: %0d" % (epoch), total=len(data_loader))
        for step, batch in enumerate(data_iter):
            inputs, labels = (batch, batch)
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            outputs = self.model(inputs, None, None, labels)
            loss = outputs[0]

            loss.backward()
            # self.lr_scheduler.step()
            self.optimizer.step()
            self.optimizer.zero_grad()

            loss_item = loss.numpy().item()
            losses.update(loss_item)

            data_iter.write("Epoch:%0d, step:%000d, avg_loss:%.4f, loss:%.4f" % (epoch, step, losses.avg, losses.val))

        print("Training:%0d, avg_loss:%.4f" % (epoch, losses.avg))

    def evaluate(self, data_loader, epoch=0):
        self.model.eval()
        losses = AverageMeter("loss")

        data_iter = tqdm.tqdm(data_loader, desc="Evaluate: ", total=len(data_loader))
        for step, batch in enumerate(data_iter):
            inputs, labels = (batch, batch)
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            outputs = self.model(inputs, labels)
            loss = outputs[0]
            
            loss_item = loss.numpy().item()
            losses.update(loss_item)
            data_iter.write("Epoch:%0d, step:%000d, avg_loss:%.4f, loss:%.4f" % (epoch, step, losses.avg, losses.val))

        print("Evaluating:%0d, avg_loss:%.4f" % (epoch, losses.avg))


    def save(self, epoch, file_path="checkpoints"):
        """
        Saving the current model on file_path

        :param epoch: current epoch number
        :param file_path: model output path which gonna be file_path+"ep%d" % epoch
        :return: final_output_path
        """
        output_path = file_path + "epoch%d" % epoch
        flow.save(self.model.state_dict(), output_path)
        print("EP:%d Model Saved on:" % epoch, output_path)
        return output_path
