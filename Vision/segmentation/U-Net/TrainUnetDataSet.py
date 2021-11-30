import oneflow as flow
import oneflow.optim
from oneflow.utils.data import Dataset
from oneflow import optim, utils
from dataloader import SelfDataSet
from log import Logger
import os
import oneflow.nn as nn
import sys
from unet import UNet
import time
import argparse


def _parse_args():
    parser = argparse.ArgumentParser("Flags for train U-Net")
    parser.add_argument(
        "--data_path", type=str, default="train_image", help="data_path"
    )
    parser.add_argument("--epochs", type=int, default=40, help="epochs")
    parser.add_argument("--batch_size", type=int, default=1, help="batch_size")
    return parser.parse_args()


Unet_train_txt = Logger('Unet_train.txt')


def Train_Unet(net, device, data_path, batch_size=3, epochs=40, lr=0.0001):
    train_dataset = SelfDataSet(data_path)
    train_loader = utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True)

    opt = optim.Adam((net.parameters()))
    loss_fun = nn.BCEWithLogitsLoss()
    bes_los = float('inf')

    for epoch in range(epochs):
        net.train()
        running_loss = 0.0
        i = 0
        begin = time.perf_counter()
        for image, label in train_loader:
            opt.zero_grad()
            image = image.to(device=device, dtype=flow.float32)
            label = label.to(device=device, dtype=flow.float32)
            pred = net(image)
            loss = loss_fun(pred, label)
            loss.backward()
            i = i + 1
            running_loss = running_loss+loss.item()
            opt.step()
        end = time.perf_counter()
        loss_avg_epoch = running_loss/i
        Unet_train_txt.write(str(format(loss_avg_epoch, '.4f')) + '\n')
        print('epoch: %d avg loss: %f time:%d s' %
              (epoch, loss_avg_epoch, end - begin))
        if loss_avg_epoch < bes_los:
            bes_los = loss_avg_epoch
            state = {'net': net.state_dict(), 'opt': opt.state_dict(),
                     'epoch': epoch}
            flow.save(state, './checkpoints')


def main(args):
    DEVICE = "cuda" if flow.cuda.is_available() else "cpu"
    print("Using {} device".format(DEVICE))
    net = UNet(1, 1,  bilinear=False)
    # print(net)
    net.to(device=DEVICE)
    data_path = args.data_path
    Train_Unet(net, DEVICE, data_path, epochs=args.epochs,
               batch_size=args.batch_size)
    Unet_train_txt.close()


if __name__ == '__main__':
    args = _parse_args()
    main(args)
