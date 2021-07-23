import argparse
import time
import numpy as np
from tsn.datasets.dataset import TSNDataSet
from tsn.models.TSN import TSN
from tsn.utils.checkpoint import load_checkpoint
import oneflow.experimental as flow

from tsn.datasets.transform import *

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

def parse_args():
    parser = argparse.ArgumentParser(description='Test an action recognizer')
    parser.add_argument('--test_mode', default=True,
                        help='train or test mode')
    parser.add_argument('--checkpoint', default='/home/liling/work/oneflow/TSN/weights/tsn_model_oneflow', help='checkpoint file')
    parser.add_argument('--data_dir',
                        default='/home/liling/work/oneflow/TSN/data',
                        help='data file path')
    parser.add_argument('--batch_size',
                        default=2,
                        help='batch_size')
    parser.add_argument('--normvalue', default={'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225], 'to_rgb': True},
                        help='data normalization value')
    parser.add_argument('--input_mean',
                        default=[0.485, 0.456, 0.406],
                        help='data normalization value')
    parser.add_argument('--input_std',
                        default=[0.229, 0.224, 0.225],
                        help='data normalization value')
    #for spatial_temporal module
    parser.add_argument('--spatial_type', default='avg',
                        help='data normalization value')
    parser.add_argument('--spatial_size', default=7,
                        help='data normalization value')
    #for segmental consensus
    parser.add_argument('--consensus_type', default='avg',
                        help='data normalization value')
    #for class head
    parser.add_argument('--spatial_feature_size', default=1,
                        help='data normalization value')
    parser.add_argument('--dropout_ratio', default=0.0,
                        help='data normalization value')
    parser.add_argument('--num_classes', default=400,
                        help='data normalization value')
    parser.add_argument('--out', help='output result file', default='default.pkl')
    parser.add_argument('--use_softmax', action='store_true',
                        help='whether to use softmax score')
    # only for TSN3D
    parser.add_argument('--fcn_testing', action='store_true',
                        help='use fcn testing for 3D convnet')
    parser.add_argument('--local_rank', type=int, default=0)

    args = parser.parse_args()
    return args

def top_k_hit(score, lb_set, k=3):
    idx = np.argsort(score)[-k:]
    return len(lb_set.intersection(idx)) > 0, 1

def top_k_accuracy(scores, labels, k=(1,)):
    res = []
    for kk in k:
        hits = []
        for x, y in zip(scores, labels):
            y = [y] if isinstance(y, int) else y
            hits.append(top_k_hit(x, set(y), k=kk)[0])
        res.append(np.mean(hits))
    return res

def multi_test(model, data_loader):
    global args
    model.eval()
    model.to("cuda")
    results = []
    rank = 0
    count = 0
    data_time_pool = 0
    proc_time_pool = 0
    tic = time.time()

    for idx in range(int(len(data_loader)/args.batch_size)):
        data, label = data_loader[idx]
        if idx % 100 == 0:
            print('rank {}, data_batch {}'.format(rank, idx))
        count = count + 1
        tac = time.time()
        data_time_pool = data_time_pool + tac - tic

        num_modalities = 1
        img_meta = label
        img_group = data.reshape(args.batch_size, int(data.shape[0]/args.batch_size), data.shape[1], data.shape[2], data.shape[3])
        img_group = flow.Tensor(img_group, device=flow.device("cuda"))

        with flow.no_grad():
            result = model(num_modalities, img_meta, img_group)
        results.append(result)

        toc = time.time()
        proc_time_pool = proc_time_pool + toc - tac

        tic = toc
    return results

def main():
    flow.env.init()
    flow.enable_eager_execution()

    global args
    args = parse_args()

    if args.out is not None and not args.out.endswith(('.pkl', '.pickle')):
        raise ValueError('The output file must be a pkl file.')

    ann_file = args.data_dir + "/kinetics400/kinetics400_val_list_videos_sub_frames.txt"
    prefix = args.data_dir + "/kinetics400/rawframes_val"

    dataset = TSNDataSet("", ann_file, num_segments=3,
                   video_dir=prefix,
                   new_length=1,
                   image_tmpl='img_{:05d}.jpg',
                   test_mode=True,
                   sample=GroupOverSample(224, 256),
                   stack = Stack(roll=False),
                   Normalize = GroupNormalize(args.input_mean, args.input_std),
                   batch_size = args.batch_size
                   )

    model=TSN(args.spatial_feature_size,
              args.dropout_ratio,
              args.num_classes)

    load_checkpoint(model, args.checkpoint)

    outputs = multi_test(model, dataset)

    results = []
    for res in outputs:
        for idx in range(0, args.batch_size):
            results.append(res[idx])

    gt_labels = []
    for i in range(len(dataset)):
        ann = dataset.get_ann_info(i)
        gt_labels.append(ann['label'])

    top1, top5 = top_k_accuracy(results, gt_labels, k=(1, 5))
    print("Top-1 Accuracy = {:.02f}".format(top1 * 100))
    print("Top-5 Accuracy = {:.02f}".format(top5 * 100))

if __name__ == '__main__':
    main()
