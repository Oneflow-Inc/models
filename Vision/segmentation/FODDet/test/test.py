import os
import argparse
import numpy as np
from dataloader import get_datadir_path, get_test_augmentation, Dataset
from model.UNet import UNet
from visualize import visualize
import oneflow as flow
from tqdm import tqdm


def _parse_args():
    parser = argparse.ArgumentParser("flags for test FODDet")
    parser.add_argument(
        "--pretrained_path",
        type=str,
        default="./checkpoints",
        help="save checkpoint root dir",
    )
    parser.add_argument("--data_dir", type=str, default="./CamVid", help="dataset path")
    parser.add_argument(
        "--save_path", type=str, default="./test_results", help="val batch size"
    )

    return parser.parse_args()


def main(args):
    if not os.path.exists(args.save_path):
        os.mkdir(args.save_path)

    net = UNet(n_channels=3, n_classes=1)

    checkpoint = flow.load(args.pretrained_path)
    net.load_state_dict(checkpoint)

    net.to("cuda")

    x_test_dir, y_test_dir = get_datadir_path(args, split="test")

    test_dataset = Dataset(
        x_test_dir, y_test_dir, augmentation=get_test_augmentation(),
    )

    print("Begin Testing...")
    for i, (image, mask) in enumerate(tqdm(test_dataset)):
        show_image = image
        with flow.no_grad():
            image = image / 255.0
            image = image.astype(np.float32)
            image = flow.tensor(image, dtype=flow.float32)
            image = image.permute(2, 0, 1)
            image = image.to("cuda")

            pred = net(image.unsqueeze(0).to("cuda"))
            pred = pred.numpy()
            pred = pred > 0.5
        save_picture_name = os.path.join(args.save_path, "test_image_" + str(i))
        visualize(
            save_picture_name, image=show_image, GT=mask[0, :, :], Pred=pred[0, 0, :, :]
        )


if __name__ == "__main__":
    args = _parse_args()
    main(args)
