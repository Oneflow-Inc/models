import argparse
import time
import numpy as np
from tsn.datasets.video_dataset import VideoDataset
from tsn.datasets.build_loader import build_dataloader
from tsn.models.TSN import TSN
from tsn.utils.checkpoint import load_checkpoint
import oneflow.experimental as flow

import warnings

warnings.filterwarnings("ignore", category=UserWarning)


def parse_args():
    parser = argparse.ArgumentParser(description="Test an action recognizer")
    parser.add_argument("--test_mode", default=True, help="train or test mode")
    parser.add_argument(
        "--checkpoint",
        default="/home/liling/work/oneflow/TSN/weights/tsn_model_oneflow",
        help="checkpoint file",
    )
    parser.add_argument("--data_dir", default="../data", help="data file path")
    # parser.add_argument('--annfile', default='/home/lil/workspace/TSN/data/kinetics400/kinetics400_val_list_videos_sub.txt',
    #                     help='annotation file path')
    # parser.add_argument('--prefix', default='/home/lil/workspace/TSN/data/kinetics400/videos_val',
    #                     help='test videos prefix path')
    parser.add_argument(
        "--normvalue",
        default={
            "mean": [123.675, 116.28, 103.53],
            "std": [58.395, 57.12, 57.375],
            "to_rgb": True,
        },
        help="data normalization value",
    )
    # for spatial_temporal module
    parser.add_argument(
        "--spatial_type", default="avg", help="data normalization value"
    )
    parser.add_argument("--spatial_size", default=7, help="data normalization value")
    # for segmental consensus
    parser.add_argument(
        "--consensus_type", default="avg", help="data normalization value"
    )
    # for class head
    parser.add_argument(
        "--spatial_feature_size", default=1, help="data normalization value"
    )
    parser.add_argument("--dropout_ratio", default=0.4, help="data normalization value")
    parser.add_argument("--num_classes", default=400, help="data normalization value")
    parser.add_argument("--out", help="output result file", default="default.pkl")
    parser.add_argument(
        "--use_softmax", action="store_true", help="whether to use softmax score"
    )
    # only for TSN3D
    parser.add_argument(
        "--fcn_testing", action="store_true", help="use fcn testing for 3D convnet"
    )
    parser.add_argument("--local_rank", type=int, default=0)

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
    for i, data in enumerate(data_loader):
        if i % 100 == 0:
            print("rank {}, data_batch {}".format(rank, i))
        count = count + 1
        tac = time.time()
        data_time_pool = data_time_pool + tac - tic

        num_modalities = int(data["num_modalities"].data[0][0])
        # print(num_modalities)
        img_meta = data["img_meta"].data
        # print(data['img_group_0'].data[0].shape)
        img_group = data["img_group_0"].data[0].numpy()
        img_group = flow.Tensor(img_group, device=flow.device("cuda"))
        # print(img_group)
        result = model(num_modalities, img_meta, img_group)
        results.append(result)

        toc = time.time()
        proc_time_pool = proc_time_pool + toc - tac

        tic = toc
    # print('rank {}, begin collect results'.format(rank), flush=True)
    # results = collect_results(results, len(data_loader.dataset), tmpdir)
    return results


def main():
    flow.env.init()
    flow.enable_eager_execution()

    global args
    args = parse_args()

    if args.out is not None and not args.out.endswith((".pkl", ".pickle")):
        raise ValueError("The output file must be a pkl file.")

    ann_file = args.data_dir + "/kinetics400/kinetics400_val_list_videos_sub.txt"
    prefix = args.data_dir + "/kinetics400/videos_val"
    dataset = VideoDataset(
        ann_file=ann_file,
        img_prefix=prefix,
        img_norm_cfg=args.normvalue,
        num_segments=3,
        random_shift=False,
        image_tmpl="img_{:05d}.jpg",
        flip_ratio=0,
        oversample="ten_crop",
        test_mode=True,
    )

    data_loader = build_dataloader(
        dataset, imgs_per_gpu=1, workers_per_gpu=1, dist=False, shuffle=False
    )

    model = TSN(args.spatial_feature_size, args.dropout_ratio, args.num_classes)

    load_checkpoint(model, args.checkpoint)
    # params = model.state_dict()
    # for key, value in params.items():
    #     if key=="backbone.conv1.weight":
    #         print(key)
    #         print(value)

    outputs = multi_test(model, data_loader)

    # print("Averaging score over {} clips without softmax (ie, raw)".format(outputs[0].shape[0]))
    results = [res.mean(axis=0) for res in outputs]
    # print(results[0])

    gt_labels = []
    for i in range(len(dataset)):
        ann = dataset.get_ann_info(i)
        gt_labels.append(ann["label"])

    top1, top5 = top_k_accuracy(results, gt_labels, k=(1, 5))
    print("Top-1 Accuracy = {:.02f}".format(top1 * 100))
    print("Top-5 Accuracy = {:.02f}".format(top5 * 100))


if __name__ == "__main__":
    main()
