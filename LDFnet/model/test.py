#!/usr/bin/python3
# coding=utf-8

import os
import sys

sys.path.insert(0, '../')
sys.dont_write_bytecode = True

import sys
import datetime
import os
import numpy as np
import dataset
from net import LDF
import oneflow as flow
import oneflow.nn as nn
from oneflow import Tensor
import cv2


class Test(object):
    def __init__(self, Dataset, Network, Path):
        ## dataset

        self.cfg = Dataset.Config(datapath=Path, snapshot='./out/LDF_model', mode='test')
        self.data = Dataset.Data(self.cfg)
        ## network
        self.net = Network(self.cfg)
        self.net.train(False)
        self.net.to('cuda')

    def save_body_detail(self):
        with flow.no_grad():
            for i in range(len(self.data.samples)):
                image, shape, name = self.data[i]
                image = image[np.newaxis, :]
                image = flow.Tensor(image).to('cuda')

                outb1, outd1, out1, outb2, outd2, out2 = self.net(image, shape)
                out = out2
                pred = flow.sigmoid(out[0, 0]).to('cpu').numpy() * 255
                head = '../eval/maps/' + self.cfg.datapath.split('/')[-1]
                if not os.path.exists(head):
                    os.makedirs(head)
                cv2.imwrite(head + '/' + name + '.png', np.round(pred))
                print(head + '/' + name + '.png')


if __name__ == '__main__':
    t = Test(dataset, LDF, '../data/DUTS-TE')
    t.save_body_detail()
