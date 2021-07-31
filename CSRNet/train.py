import os
from dataset import listDataset
from model import CSRNet
from utils import save_checkpoint
import argparse
import json
import time
import oneflow as flow
import oneflow.nn as nn
import transforms.spatial_transforms as ST

parser = argparse.ArgumentParser(description='PyTorch CSRNet')
parser.add_argument('train_json', metavar='TRAIN',
                    help='path to train json')
parser.add_argument('test_json', metavar='TEST',
                    help='path to test json')

parser.add_argument('--pre', '-p', metavar='PRETRAINED', default=None, type=str,
                    help='path to the pretrained model')

parser.add_argument('gpu', metavar='GPU', type=str,
                    help='GPU id to use.')

parser.add_argument('task', metavar='TASK', type=str,
                    help='task id to use.')


def main():
    global args, best_prec1
    best_prec1 = 1e6
    args = parser.parse_args()
    args.original_lr = 1e-7
    args.lr = 1e-7
    args.batch_size = 1
    args.momentum = 0.95
    args.decay = 5 * 1e-4
    args.start_epoch = 0
    args.epochs = 400
    args.steps = [-1, 1, 100, 150]
    args.scales = [1, 1, 1, 1]
    args.workers = 0
    args.seed = time.time()
    args.print_freq = 30
    with open(args.train_json, 'r') as outfile:
        train_list = json.load(outfile)
    with open(args.test_json, 'r') as outfile:
        val_list = json.load(outfile)


    # 设置gpu
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    model = CSRNet()
    model = model.to("cuda")
    criterion = nn.MSELoss(reduction="sum").to("cuda")
    optimizer = flow.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,   weight_decay=args.decay)


    if args.pre:
        if os.path.isfile(args.pre):
            print("=> loading checkpoint '{}'".format(args.pre))
            checkpoint = flow.load(args.pre)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            #7.21尝试oneflow版本能否加载
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.pre, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.pre))

    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch)

        train(train_list, model, criterion, optimizer, epoch)
        prec1 = validate(val_list, model, criterion)
        #prec1 = 0


        is_best = prec1 < best_prec1
        best_prec1 = min(prec1, best_prec1)
        print(' * best MAE {mae:.3f} '
              .format(mae=best_prec1))
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.pre,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
            #'optimizer': optimizer.state_dict(),
        }, is_best, str(epoch+1))


def train(train_list, model, criterion, optimizer, epoch):
    losses = AverageMeter()
    batch_time = AverageMeter()
    data_time = AverageMeter()


    train_loader = listDataset(train_list,
                       shuffle=True,
                       transform=ST.Compose([
                                    ST.ToNumpy(),ST.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
                   ]),
                       train=True,
                       seen=model.seen,
                       batch_size=args.batch_size,
                       num_workers=args.workers)
    model.train()
    end = time.time()
    for i, (img, target) in enumerate(train_loader):
        data_time.update(time.time() - end)
        img=flow.Tensor(img, dtype=flow.float32, device="cuda")
        output = model(img).to("cuda")
        output=flow.Tensor(output,device="cuda")
        target = flow.Tensor(target, device="cuda").unsqueeze(0)
        loss = criterion(output, target)
        losses.update(loss.numpy()[0], img.size(0))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        batch_time.update(time.time() - end)
        end = time.time()
        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                .format(
                epoch, i, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses))


def validate(val_list, model, criterion):
    print('begin test')
    test_loader = listDataset(val_list,
                               shuffle=True,
                               transform=ST.Compose([
                                   ST.ToNumpy(), ST.Normalize(mean=[0.485, 0.456, 0.406],
                                                              std=[0.229, 0.224, 0.225]),
                               ]),
                               train=True,
                               seen=model.seen,
                               batch_size=args.batch_size,
                               num_workers=args.workers)

    model.eval()
    mae = 0
    for i, (img, target) in enumerate(test_loader):
        img = flow.Tensor(img, dtype=flow.float32, device="cuda")
        with flow.no_grad():
            output = model(img).to("cuda")


       # output = model(img).to("cuda")
       # output = flow.Tensor(output, device="cuda")
        mae += abs(output.data.sum().numpy()[0] - target.sum())



    mae = mae / len(test_loader)
    print(' * MAE {mae:.3f} '
          .format(mae=mae))
    return mae

def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""

    args.lr = args.original_lr

    for i in range(len(args.steps)):
        scale = args.scales[i] if i < len(args.scales) else 1
        if epoch >= args.steps[i]:
            args.lr = args.lr * scale
            if epoch == args.steps[i]:
                break
        else:
            break
    for param_group in optimizer.param_groups:
        param_group['lr'] = args.lr


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        # 方差
        self.val = 0
        # 平均数
        self.avg = 0
        # 和
        self.sum = 0
        # 数量
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

if __name__ == '__main__':

    main()
