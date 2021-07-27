import argparse
import time
import os
import numpy as np
from tsn.models.TSN import TSN
from tsn.utils.checkpoint import load_checkpoint
import oneflow as flow
from tsn.datasets.transform import *
from tsn.datasets.dataset import TSNDataSet

import warnings

warnings.filterwarnings("ignore", category=UserWarning)


def parse_args():
    parser = argparse.ArgumentParser(description="Test an action recognizer")
    parser.add_argument("--test_mode", default=False, help="train or test mode")
    parser.add_argument("-load_checkpoint", default="", help="checkpoint file")
    parser.add_argument(
        "--data_dir", default="./data", help="data file path",
    )
    parser.add_argument(
        "--pretrained",
        default="./resnet50_imagenet_pretrain_model",
        help="test videos prefix path",
    )
    parser.add_argument(
        "--input_mean", default=[0.485, 0.456, 0.406], help="data normalization value"
    )
    parser.add_argument(
        "--input_std", default=[0.229, 0.224, 0.225], help="data normalization value"
    )
    parser.add_argument("--epochs", default=100, help="max epochs for training")
    parser.add_argument("--lr_steps", default=[40, 80], help="lr step")
    parser.add_argument("--batch_size", default=4, help="imgs per gpu")
    parser.add_argument(
        "--save_checkpoint_path", default="./res", help="imgs per gpu",
    )
    parser.add_argument("--lr", default=0.0025, help="learning_rate")
    parser.add_argument("--mom", default=0.9, help="momentum")
    parser.add_argument("--weight_decay", default=0.000, help="weight decay")
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
        param_group["lr"] = lr


def main():
    
    flow.InitEagerGlobalSession()

    global args
    args = parse_args()

    model = TSN(
        args.spatial_feature_size, args.dropout_ratio, args.num_classes, args.pretrained
    )

    train_ann_file = (
        args.data_dir + "/kinetics400/kinetics400_train_list_videos_sub_frames.txt"
    )
    train_prefix = args.data_dir + "/kinetics400/rawframes_train"
    val_ann_file = (
        args.data_dir + "/kinetics400/kinetics400_val_list_videos_sub_frames.txt"
    )
    val_prefix = args.data_dir + "/kinetics400/rawframes_val"

    train_dataset = TSNDataSet(
        "",
        train_ann_file,
        num_segments=3,
        video_dir=train_prefix,
        new_length=1,
        image_tmpl="img_{:05d}.jpg",
        flip=GroupMultiScaleCrop(224, [1, 0.875, 0.75, 0.66]),
        crop=GroupRandomHorizontalFlip(is_flow=False),
        stack=Stack(roll=False),
        Normalize=GroupNormalize(args.input_mean, args.input_std),
        batch_size=args.batch_size,
    )

    val_dataset = TSNDataSet(
        "",
        val_ann_file,
        num_segments=3,
        video_dir=val_prefix,
        new_length=1,
        image_tmpl="img_{:05d}.jpg",
        test_mode=True,
        sample=GroupOverSample(224, 256),
        stack=Stack(roll=False),
        Normalize=GroupNormalize(args.input_mean, args.input_std),
        batch_size=args.batch_size,
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

        for idx in range(int(len(train_dataset) / args.batch_size)):
            data, label = train_dataset[idx]
            num_modalities = 1
            img_group = data.reshape(
                args.batch_size,
                int(data.shape[0] / args.batch_size),
                data.shape[1],
                data.shape[2],
                data.shape[3],
            )
            start_t = time.time()
            img_group = flow.Tensor(img_group, device=flow.device("cuda"))
            gt_label = flow.Tensor(label, device=flow.device("cuda"))
            gt_label = gt_label.squeeze().to(dtype=flow.int32)

            predeicts = model(num_modalities, gt_label, img_group, True)
            loss = of_corss_entropy(predeicts, gt_label)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            end_t = time.time()

            if idx % print_interval == 0:
                l = loss.numpy()[0]
                losses.append(l)
                tmp_lr = optimizer.param_groups[0]["lr"]
                print(
                    "epoch {} train iter {} oneflow loss {}, lr: {}, train time : {}".format(
                        epoch, idx, l, tmp_lr, end_t - start_t
                    )
                )

        if epoch % val_interval == 0:
            model.eval()
            results = []

            for idx in range(int(len(val_dataset) / args.batch_size)):
                data, label = val_dataset[idx]
                num_modalities = 1
                img_meta = label
                img_group = data.reshape(
                    args.batch_size,
                    int(data.shape[0] / args.batch_size),
                    data.shape[1],
                    data.shape[2],
                    data.shape[3],
                )
                img_group = flow.Tensor(img_group, device=flow.device("cuda"))
                with flow.no_grad():
                    result = model(num_modalities, img_meta, img_group)
                results.append(result)

            outputs = []
            for res in results:
                for idx in range(0, args.batch_size):
                    outputs.append(res[idx])

            gt_labels = []
            for i in range(len(val_dataset)):
                ann = val_dataset.get_ann_info(i)
                gt_labels.append(ann["label"])

            top1, top5 = top_k_accuracy(outputs, gt_labels, k=(1, 5))
            print(
                "epoch %d, oneflow top1 val acc: %f, top5 val acc: %f"
                % (epoch, top1 * 100, top5 * 100)
            )
            flow.save(
                model.state_dict(),
                os.path.join(
                    args.save_checkpoint_path,
                    "epoch_%d_val_acc_%f_%f" % (epoch, top1, top5),
                ),
            )

    writer = open("of_losses.txt", "w")
    for o in losses:
        writer.write("%f\n" % o)
    writer.close()


if __name__ == "__main__":
    main()
