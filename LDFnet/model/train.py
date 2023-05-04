#!/usr/bin/python3
# coding=utf-8

import sys
import datetime
import os
import numpy as np
import dataset
from net import LDF
import oneflow as flow
import oneflow.nn as nn
from oneflow import Tensor

sys.path.insert(0, '../')
sys.dont_write_bytecode = True

load_weights_dir = "./pretrained/resnet50-19c8e357"

os.environ["CUDA_VISIBLE_DEVICES"] = '1'


def iou_loss(pred, mask):
    pred = flow.sigmoid(pred)
    inter = (pred * mask).sum(dim=(2, 3))
    union = (pred + mask).sum(dim=(2, 3))
    iou = 1 - (inter + 1) / (union - inter + 1)
    return iou.mean()


def train(Dataset, Network):
    ## dataset
    cfg = Dataset.Config(datapath='../data/DUTS-TR', savepath='./out', mode='train', batch=8, lr=0.05, momen=0.9,
                         decay=5e-4, epoch=48)
    data = Dataset.Data(cfg)
    net = Network(cfg)
    net.to(flow.device('cuda'))

    ## parameter
    base, head = [], []

    for name, param in net.named_parameters():
        # print(name)
        if 'model.conv1' in name or 'model.bn1' in name:
            print(name)
        elif 'model' in name:
            base.append(param)
        else:
            head.append(param)

    optimizer = flow.optim.SGD([{'params': base}, {'params': head}], lr=cfg.lr, momentum=cfg.momen,
                               weight_decay=cfg.decay)
    bce = flow.nn.BCEWithLogitsLoss()
    bce.to(flow.device('cuda'))

    global_step = 0
    batch_num = len(data) // cfg.batch
    best_loss = 9999999
    for epoch in range(cfg.epoch):
        net.train()
        optimizer.param_groups[0]['lr'] = (1 - abs((epoch + 1) / (cfg.epoch + 1) * 2 - 1)) * cfg.lr * 0.1
        optimizer.param_groups[1]['lr'] = (1 - abs((epoch + 1) / (cfg.epoch + 1) * 2 - 1)) * cfg.lr
        temp_loss = 0
        for batch_idx in range(batch_num):
            image, mask, body, detail = [], [], [], []
            for idx in range(cfg.batch):
                im, ma, bo, de = data[batch_idx * cfg.batch + idx]
                image.append(im[np.newaxis, :])
                mask.append(ma[np.newaxis, :])
                body.append(bo[np.newaxis, :])
                detail.append(de[np.newaxis, :])

            image = np.ascontiguousarray(np.concatenate(image, axis=0).astype(np.float32))
            mask = np.ascontiguousarray(np.concatenate(mask, axis=0).astype(np.float32)).reshape((8, 1, 352, 352))
            body = np.ascontiguousarray(np.concatenate(body, axis=0).astype(np.float32)).reshape((8, 1, 352, 352))
            detail = np.ascontiguousarray(np.concatenate(detail, axis=0).astype(np.float32)).reshape((8, 1, 352, 352))

            image = flow.Tensor(image).to(flow.device('cuda'))
            mask = flow.Tensor(mask).to(flow.device('cuda'))
            body = flow.Tensor(body).to(flow.device('cuda'))
            detail = flow.Tensor(detail).to(flow.device('cuda'))

            outb1, outd1, out1, outb2, outd2, out2 = net(image)

            lossb1 = bce(outb1, body)
            lossd1 = bce(outd1, detail)
            loss1 = bce(out1, mask) + iou_loss(out1, mask)

            lossb2 = bce(outb2, body)
            lossd2 = bce(outd2, detail)
            loss2 = bce(out2, mask) + iou_loss(out2, mask)
            loss = flow.Tensor((lossb1 + lossd1 + loss1 + lossb2 + lossd2 + loss2) / 2).to(flow.device('cuda'))
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            global_step += 1
            temp_loss += loss.numpy()
            if batch_idx % 10 == 0:
                print(
                    '%s | step:%d/%d/%d | lr=%.6f | lossb1=%.6f | lossd1=%.6f | loss1=%.6f | lossb2=%.6f | lossd2=%.6f | loss2=%.6f'
                    % (datetime.datetime.now(), global_step, epoch + 1, cfg.epoch, optimizer.param_groups[0]['lr'],
                       lossb1.numpy(), lossd1.numpy(), loss1.numpy(), lossb2.numpy(), lossd2.numpy(), loss2.numpy()))

        temp_loss = temp_loss / batch_num
        if temp_loss < best_loss:
            best_loss = temp_loss
            print('****************now saving the pretrained model - ', epoch + 1)
            flow.save(net.state_dict(), cfg.savepath + '/model-' + str(epoch + 1)+str(best_loss))


if __name__ == '__main__':
    train(dataset, LDF)
