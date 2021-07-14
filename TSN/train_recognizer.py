import argparse
import time
import os
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
    parser.add_argument("--test_mode", default=False, help="train or test mode")
    parser.add_argument("-load_checkpoint", default="", help="checkpoint file")
    parser.add_argument("--data_dir", default="../data", help="data file path")
    # parser.add_argument('--train_annfile', default='/home/liling/work/oneflow/TSN/data/kinetics400/kinetics400_train_list_videos_sub.txt',
    #                     help='train annotation file path')
    # parser.add_argument('--train_prefix', default='/home/liling/work/oneflow/TSN/data/kinetics400/videos_train',
    #                     help='train videos prefix path')
    # parser.add_argument('--val_annfile',
    #                     default='/home/liling/work/oneflow/TSN/data/kinetics400/kinetics400_val_list_videos_sub.txt',
    #                     help='validation annotation file path')
    # parser.add_argument('--val_prefix', default='/home/liling/work/oneflow/TSN/data/kinetics400/videos_val',
    #                     help='test videos prefix path')
    parser.add_argument(
        "--pretrained",
        default="/home/liling/work/oneflow/TSN/weights/resnet50_imagenet_pretrain_model",
        help="test videos prefix path",
    )
    parser.add_argument(
        "--normvalue",
        default={
            "mean": [123.675, 116.28, 103.53],
            "std": [58.395, 57.12, 57.375],
            "to_rgb": True,
        },
        help="data normalization value",
    )
    parser.add_argument("--epochs", default=100, help="max epochs for training")
    parser.add_argument("--lr_steps", default=[40, 80], help="lr step")
    parser.add_argument("--imgs_per_gpu", default=8, help="imgs per gpu")
    parser.add_argument(
        "--save_checkpoint_path",
        default="/home/liling/work/oneflow/TSN/res",
        help="imgs per gpu",
    )
    parser.add_argument("--lr", default=0.0025, help="learning_rate")
    parser.add_argument("--mom", default=0.9, help="momentum")
    parser.add_argument("--weight_decay", default=0.0000, help="weight decay")
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


def adjust_learning_rate(optimizer, epoch, lr_steps):
    """Sets the learning rate to the initial LR decayed by lr steps"""
    decay = 0.1 ** (sum(epoch >= np.array(lr_steps)))
    lr = args.lr * decay

    for param_group in optimizer.param_groups:
        # param_group['lr'] = lr * param_group['lr']
        param_group["lr"] = lr


def main():
    flow.enable_eager_execution()
    flow.InitEagerGlobalSession()

    global args
    args = parse_args()

    model = TSN(
        args.spatial_feature_size, args.dropout_ratio, args.num_classes, args.pretrained
    )

    train_ann_file = (
        args.data_dir + "/kinetics400/kinetics400_train_list_videos_sub.txt"
    )
    train_prefix = args.data_dir + "/kinetics400/videos_train"
    val_ann_file = args.data_dir + "/kinetics400/kinetics400_val_list_videos_sub.txt"
    val_prefix = args.data_dir + "/kinetics400/videos_val"
    train_dataset = VideoDataset(
        ann_file=train_ann_file,
        img_prefix=train_prefix,
        img_norm_cfg=args.normvalue,
        image_tmpl="img_{:05d}.jpg",
        multiscale_crop=True,
        scales=[1, 0.875, 0.75, 0.66],
        test_mode=False,
    )

    val_dataset = VideoDataset(
        ann_file=val_ann_file,
        img_prefix=val_prefix,
        img_norm_cfg=args.normvalue,
        num_segments=3,
        random_shift=False,
        image_tmpl="img_{:05d}.jpg",
        flip_ratio=0,
        oversample="ten_crop",
        test_mode=True,
    )

    train_data_loader = build_dataloader(
        train_dataset,
        imgs_per_gpu=args.imgs_per_gpu,
        workers_per_gpu=1,
        num_gpus=1,
        dist=False,
    )

    val_data_loader = build_dataloader(
        val_dataset, imgs_per_gpu=1, workers_per_gpu=1, dist=False, shuffle=False
    )

    if args.load_checkpoint != "":
        model.load_state_dict(flow.load(args.load_checkpoint))

    of_corss_entropy = flow.nn.CrossEntropyLoss()
    of_corss_entropy.to("cuda")

    model.to("cuda")

    optimizer = flow.optim.SGD(model.parameters(), lr=args.lr, momentum=args.mom)

    losses = []
    print_interval = 50
    val_interval = 1
    for epoch in range(args.epochs):
        adjust_learning_rate(optimizer, epoch, args.lr_steps)
        model.train()

        for i, data_batch in enumerate(train_data_loader):
            num_modalities = int(data_batch["num_modalities"].data[0][0])
            # print(num_modalities)
            gt_label = data_batch["gt_label"].data[0].numpy()
            # print(gt_label)
            img_group = data_batch["img_group_0"].data[0].numpy()
            start_t = time.time()
            img_group = flow.Tensor(img_group, device=flow.device("cuda"))
            gt_label = flow.Tensor(gt_label, device=flow.device("cuda"))
            gt_label = gt_label.squeeze()
            gt_label = gt_label.to(dtype=flow.int32)
            predeicts = model(num_modalities, gt_label, img_group, True)
            loss = of_corss_entropy(predeicts, gt_label)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            end_t = time.time()

            if i % print_interval == 0:
                l = loss.numpy()[0]
                losses.append(l)
                tmp_lr = optimizer.param_groups[0]["lr"]
                print(
                    "epoch {} train iter {} oneflow loss {}, lr: {}, train time : {}".format(
                        epoch, i, l, tmp_lr, end_t - start_t
                    )
                )

        if epoch % val_interval == 0:
            model.eval()
            results = []
            for i, data in enumerate(val_data_loader):
                num_modalities = int(data["num_modalities"].data[0][0])
                # print(num_modalities)
                img_meta = data["img_meta"].data
                # print(data['img_group_0'].data[0].shape)
                img_group = data["img_group_0"].data[0].numpy()
                img_group = flow.Tensor(img_group, device=flow.device("cuda"))
                # print(img_group)
                with flow.no_grad():
                    result = model(num_modalities, img_meta, img_group)
                    # print(result)
                    results.append(result)

            results = [res.mean(axis=0) for res in results]
            # print(results[0])

            gt_labels = []
            for i in range(len(val_data_loader)):
                ann = val_dataset.get_ann_info(i)
                gt_labels.append(ann["label"])

            top1, top5 = top_k_accuracy(results, gt_labels, k=(1, 5))
            print(
                "epoch %d, oneflow top1 val acc: %f, top5 val acc: %f"
                % (epoch, top1, top5)
            )
            flow.save(
                model.state_dict(),
                os.path.join(
                    args.save_checkpoint_path, "epoch_%d_val_acc_%f" % (epoch, top1)
                ),
            )
            # print("Top-1 Accuracy = {:.02f}".format(top1 * 100))
            # print("Top-5 Accuracy = {:.02f}".format(top5 * 100))

    writer = open("of_losses.txt", "w")
    for o in losses:
        writer.write("%f\n" % o)
    writer.close()


if __name__ == "__main__":
    main()
