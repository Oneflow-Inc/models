import ipdb
import oneflow as flow
from oneflow import nn

import argparse
import os
import shutil
import time

import numpy as np

from models.resnet50 import resnet50
from utils.ofrecord_data_utils import OFRecordDataLoader


def to_python_float(t):
    if hasattr(t, 'item'):
        return t.item()
    else:
        return t[0]


def parse():

    parser = argparse.ArgumentParser(description="Oneflow Graph Imagenette Training")
    parser.add_argument(
        "--ofrecord-path", metavar="DIR", default="./ofrecord", help="path to dataset"
    )
    parser.add_argument("--checkpoint-path", metavar='DIR',
                        default="./of_checkpoints", help="path to checkpoint save")
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--epochs', default=90, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('-b', '--batch-size', default=256, type=int,
                        metavar='N', help='mini-batch size per process (default: 256)')
    parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                        metavar='LR', help='Initial learning rate.  Will be scaled by <global batch size>/256: args.lr = args.lr*float(args.batch_size*args.world_size)/256.  A warmup schedule will also be applied over the first 5 epochs.')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)')
    parser.add_argument('--print-freq', '-p', default=10, type=int,
                        metavar='N', help='print frequency (default: 10)')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                        help='evaluate model on validation set')
    parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                        help='use pre-trained model')

    parser.add_argument('--prof', default=-1, type=int,
                        help='Only run 10 iterations for profiling.')
    parser.add_argument('--deterministic', action='store_true')

    parser.add_argument(
        "--local_rank", default=os.getenv('LOCAL_RANK', 0), type=int)
    parser.add_argument('--sync_bn', action='store_true',
                        help='enabling apex sync BN.')

    # parser.add_argument('--opt-level', type=str)
    # parser.add_argument('--keep-batchnorm-fp32', type=str, default=None)
    # parser.add_argument('--loss-scale', type=str, default=None)
    # parser.add_argument('--channels-last', type=bool, default=False)
    args = parser.parse_args()
    return args


def main():
    global best_prec1, args

    args = parse()
    # print("opt_level = {}".format(args.opt_level))
    # print("keep_batchnorm_fp32 = {}".format(args.keep_batchnorm_fp32), type(args.keep_batchnorm_fp32))
    # print("loss_scale = {}".format(args.loss_scale), type(args.loss_scale))

    # print("\nCUDNN VERSION: {}\n".format(torch.backends.cudnn.version()))

    # cudnn.benchmark = True
    best_prec1 = 0
    # if args.deterministic:
    #     cudnn.benchmark = False
    #     cudnn.deterministic = True
    #     torch.manual_seed(args.local_rank)
    #     torch.set_printoptions(precision=10)

    args.distributed = False
    if 'WORLD_SIZE' in os.environ:
        args.distributed = int(os.environ['WORLD_SIZE']) > 1

    args.gpu = 0
    args.world_size = 1

    # if args.distributed:
    #     args.gpu = args.local_rank
    #     torch.cuda.set_device(args.gpu)
    #     torch.distributed.init_process_group(backend='nccl',
    #                                          init_method='env://')
    #     args.world_size = torch.distributed.get_world_size()

    # assert torch.backends.cudnn.enabled, "Amp requires cudnn backend to be enabled."

    # if args.channels_last:
    #     memory_format = torch.channels_last
    # else:
    #     memory_format = torch.contiguous_format

    # create model
    if args.pretrained:
        print("=> using pre-trained model ")
        model = resnet50(pretrained=True)
    else:
        print("=> creating model ")
        model = resnet50()

    # if args.sync_bn:
        # import apex
        # print("using apex synced BN")
        # model = apex.parallel.convert_syncbn_model(model)

    model = model.to("cuda")

    # Scale learning rate based on global batch size
    args.lr = args.lr*float(args.batch_size*args.world_size)/256.
    optimizer = flow.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    
    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().to("cuda")

    # Build nn Graph model from eager mode
    class ResNetGraph(nn.Graph):
        def __init__(self):
            super().__init__()
            self.model = model
            self.criterion = criterion
            self.add_optimizer("sgd", optimizer)

        def build(self, input, target) -> flow.Tensor:
            output = self.model(input)
            loss = self.criterion(output, target)
            loss.backward()
            return loss, output

    class ResNetEvalGraph(nn.Graph):
        def __init__(self):
            super().__init__()
            self.model = model
        
        def build(self, input: flow.Tensor) -> flow.Tensor:
            with flow.no_grad():
                output = self.model(input)
                pred = flow.softmax(output)
            return pred
    
    resnet_graph = ResNetGraph()
    resnet_eval_graph = ResNetEvalGraph()
    
    # Optionally resume from a checkpoint
    if args.resume:
        # Use a local scope to avoid dangling references
        def resume():
            if os.path.isfile(args.resume):
                print("=> loading checkpoint '{}'".format(args.resume))
                checkpoint = flow.load(args.resume)
                args.start_epoch = checkpoint['epoch']
                global best_prec1
                best_prec1 = checkpoint['best_prec1']
                model.load_state_dict(checkpoint['state_dict'])
                # optimizer.load_state_dict(checkpoint['optimizer'])
                print("=> loaded checkpoint '{}' (epoch {})"
                      .format(args.resume, checkpoint['epoch']))
            else:
                print("=> no checkpoint found at '{}'".format(args.resume))
        resume()

    train_loader = OFRecordDataLoader(
        ofrecord_root=args.ofrecord_path,
        mode="train",
        #dataset_size=94,
        #dataset_size=940,
        dataset_size=9469,
        batch_size=args.batch_size,
    )

    val_loader = OFRecordDataLoader(
        ofrecord_root=args.ofrecord_path,
        mode="val",
        dataset_size=3925,
        batch_size=args.batch_size,
    )

    # train_loader = torch.utils.data.DataLoader(
    #     train_dataset, batch_size=args.batch_size, shuffle=(
    #         train_sampler is None),
    #     num_workers=args.workers, pin_memory=True, sampler=train_sampler, collate_fn=collate_fn)

    # val_loader = torch.utils.data.DataLoader(
    #     val_dataset,
    #     batch_size=args.batch_size, shuffle=False,
    #     num_workers=args.workers, pin_memory=True,
    #     sampler=val_sampler,
    #     collate_fn=collate_fn)

    if args.evaluate:
        validate(val_loader, resnet_eval_graph)
        return

    for epoch in range(args.start_epoch, args.epochs):
        # if args.distributed:
            # train_sampler.set_epoch(epoch)

        # train for one epoch
        train(train_loader, resnet_graph, epoch)

        # evaluate on validation set
        prec1 = validate(val_loader, resnet_eval_graph)

        # remember best prec@1 and save checkpoint
        if args.local_rank == 0:
            is_best = prec1 > best_prec1
            best_prec1 = max(prec1, best_prec1)
            save_checkpoint(model.state_dict(), 
                            is_best, os.path.join(args.checkpoint_path, "checkpoint_{}".format(epoch)))
            # save_checkpoint({
            #     'epoch': epoch + 1,
            #     'arch': "resnet50",
            #     'state_dict': model.state_dict(),
            #     'best_prec1': best_prec1,
            #     # 'optimizer': optimizer.state_dict(),
            # }, is_best)


def train(train_loader, model_graph, epoch):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to train mode
    model_graph.model.train()
    end = time.time()

    # prefetcher = data_prefetcher(train_loader)
    # input, target = prefetcher.next()
    i = 0
    # while input is not None:
    for _ in range(len(train_loader)):
        i += 1

        input, target = train_loader.get_batch()
        input = input.to("cuda")
        target = target.to("cuda")
        # adjust_learning_rate(optimizer, epoch, i, len(train_loader))

        loss, output = model_graph(input, target)

        # for param in model.parameters():
        #     print(param.data.double().sum().item(), param.grad.data.double().sum().item())

        if i % args.print_freq == 0:
            # Every print_freq iterations, check the loss, accuracy, and speed.
            # For best performance, it doesn't make sense to print these metrics every
            # iteration, since they incur an allreduce and some host<->device syncs.

            # Measure accuracy
            prec1 = accuracy(output.numpy(), target.numpy())

            # Average loss and accuracy across processes for logging
            # if args.distributed:
            #     reduced_loss = reduce_tensor(loss.data)
            #     prec1 = reduce_tensor(prec1)
            #     prec5 = reduce_tensor(prec5)
            # else:
            #     reduced_loss = loss.data
            reduced_loss = loss.numpy()

            # to_python_float incurs a host<->device sync
            losses.update(to_python_float(reduced_loss), input.size(0))
            top1.update(to_python_float(prec1), input.size(0))

            batch_time.update((time.time() - end)/args.print_freq)
            end = time.time()

            if args.local_rank == 0:
                print('Epoch: [{0}][{1}/{2}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Speed {3:.3f} ({4:.3f})\t'
                      'Loss {loss.val:.10f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                          epoch, i, len(train_loader),
                          args.world_size*args.batch_size/batch_time.val,
                          args.world_size*args.batch_size/batch_time.avg,
                          batch_time=batch_time,
                          loss=losses, top1=top1))


def validate(val_loader, model_graph):
    batch_time = AverageMeter()
    # losses = AverageMeter()
    top1 = AverageMeter()

    # switch to evaluate mode
    model_graph.model.eval()

    end = time.time()

    i = 0
    for _ in range(len(val_loader)):
        i += 1

        input, target = val_loader.get_batch()
        input = input.to("cuda")
        # compute output
        output = model_graph(input)

        # measure accuracy and record loss
        prec1 = accuracy(output.numpy(), target.numpy())

        # if args.distributed:
        #     reduced_loss = reduce_tensor(loss.data)
        #     prec1 = reduce_tensor(prec1)
        #     prec5 = reduce_tensor(prec5)
        # else:
        #     reduced_loss = loss.data

        # losses.update(to_python_float(reduced_loss), input.size(0))
        top1.update(to_python_float(prec1), input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # TODO:  Change timings to mirror train().
        if args.local_rank == 0 and i % args.print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Speed {2:.3f} ({3:.3f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                      i, len(val_loader),
                      args.world_size * args.batch_size / batch_time.val,
                      args.world_size * args.batch_size / batch_time.avg,
                      batch_time=batch_time, top1=top1))

    print(' * Prec@1 {top1.avg:.3f}'.format(top1=top1))

    return top1.avg


def save_checkpoint(state, is_best, filepath):
    flow.save(state, filepath)
    if is_best:
        filepath_best = os.path.join(os.path.dirname(filepath), "model_best")
        shutil.copytree(filepath, filepath_best)


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


def adjust_learning_rate(optimizer, epoch, step, len_epoch):
    """LR schedule that should yield 76% converged accuracy with batch size 256"""
    factor = epoch // 30

    if epoch >= 80:
        factor = factor + 1

    lr = args.lr*(0.1**factor)

    """Warmup"""
    if epoch < 5:
        lr = lr*float(1 + step + epoch*len_epoch)/(5.*len_epoch)

    # if(args.local_rank == 0):
    #     print("epoch = {}, step = {}, lr = {}".format(epoch, step, lr))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target):
    """Computes the precision@k for the specified values of k"""
    batch_size = target.shape[0]
    pred = np.argmax(output, axis=1)
    # correct = pred.eq(target.reshape(1, -1).expand_as(pred))
    correct = (pred == target).astype("float32").sum(0)
    ret = correct * 100. / batch_size
    return ret


def reduce_tensor(tensor):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.reduce_op.SUM)
    rt /= args.world_size
    return rt


if __name__ == '__main__':
    main()
