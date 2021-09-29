import os
import cv2
import numpy as np
import argparse

import oneflow as flow
import oneflow.nn as nn
import albumentations as albu
import oneflow.utils.data as flow_data


def _parse_args():
    parser = argparse.ArgumentParser("flags for train FODDet")
    parser.add_argument(
        "--save_checkpoint_path",
        type=str,
        default="./checkpoints",
        help="save checkpoint root dir",
    )
    parser.add_argument(
        "--load_checkpoint", type=str, default="", help="load checkpoint"
    )
    parser.add_argument(
        "--data_dir", type=str, default="./CamVid", help="dataset path"
    )
    # training hyper-parameters
    parser.add_argument(
        "--learning_rate", type=float, default=0.001, help="learning rate"
    )
    parser.add_argument("--weight_decay", type=float, default=1e-8, help="weight decay")
    parser.add_argument("--epochs", type=int, default=50, help="training epochs")
    parser.add_argument(
        "--train_batch_size", type=int, default=8, help="train batch size"
    )
    parser.add_argument("--val_batch_size", type=int, default=32, help="val batch size")

    return parser.parse_args() 

def get_datadir_path(args, split='train'):
    assert split in ['train', 'val', 'test']
    if split == 'train':
        x_dir = os.path.join(args.data_dir, "train")
        y_dir = os.path.join(args.data_dir, "train_labels")
    elif split == 'val':
        x_dir = os.path.join(args.data_dir, "val")
        y_dir = os.path.join(args.data_dir, "valid_labels")
    elif split == 'test':
        x_dir = os.path.join(args.data_dir, "test")
        y_dir = os.path.join(args.data_dir, "test_labels") 
    return x_dir, y_dir     


class Dataset(flow.utils.data.Dataset):
    """CamVid Dataset. Read images, apply augmentation and preprocessing transformations.

    Args:
        images_dir (str): path to images folder
        masks_dir (str): path to segmentation masks folder
        class_values (list): values of classes to extract from segmentation mask
        augmentation (albumentations.Compose): data transfromation pipeline
            (e.g. flip, scale, etc.)
        preprocessing (albumentations.Compose): data preprocessing
            (e.g. noralization, shape manipulation, etc.)
    """

    def __init__(
        self, images_dir, masks_dir, augmentation=None,
    ):
        self.ids = os.listdir(images_dir)
        self.images_fps = [os.path.join(images_dir, image_id) for image_id in self.ids]
        self.masks_fps = [os.path.join(masks_dir, image_id) for image_id in self.ids]
        self.augmentation = augmentation

    def __getitem__(self, i):
        # read data
        image = cv2.imread(self.images_fps[i])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(self.masks_fps[i], 0)

        mask = mask == 17
        mask = mask.astype("float")

        # apply augmentations
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample["image"], sample["mask"]

        return image, mask.reshape(1, 320, 320)

    def __len__(self):
        return len(self.ids)


def get_training_augmentation():
    train_transform = [
        albu.HorizontalFlip(p=0.5),
        albu.Resize(height=320, width=320, always_apply=True),
        albu.ShiftScaleRotate(
            scale_limit=0.1, rotate_limit=20, shift_limit=0.1, p=1, border_mode=0
        ),
    ]
    return albu.Compose(train_transform)


def get_test_augmentation():
    train_transform = [
        albu.Resize(height=320, width=320, always_apply=True),
    ]
    return albu.Compose(train_transform)


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2), DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(
                in_channels // 2, in_channels // 2, kernel_size=2, stride=2
            )

        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        constantpad = nn.ConstantPad2d(
            [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2]
        )
        x1 = constantpad(x1)

        x = flow.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 32)
        self.down1 = Down(32, 64)
        self.down2 = Down(64, 128)
        self.down3 = Down(128, 256)
        self.down4 = Down(256, 256)
        self.up1 = Up(512, 128, bilinear)
        self.up2 = Up(256, 64, bilinear)
        self.up3 = Up(128, 32, bilinear)
        self.up4 = Up(64, 32, bilinear)
        self.outc = OutConv(32, n_classes)
        self.out = flow.sigmoid

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        logits = self.out(logits)
        return logits

def main(args):
    x_train_dir, y_train_dir = get_datadir_path(args, split='train')
    if not os.path.exists(args.save_checkpoint_path):
        os.mkdir(args.save_checkpoint_path)

    train_dataset = Dataset(
        x_train_dir, y_train_dir, augmentation=get_training_augmentation(),
    )
    batch_size = args.train_batch_size
    train_loader = flow_data.DataLoader(train_dataset, batch_size, shuffle=True)

    net = UNet(n_channels=3, n_classes=1)
    net.to("cuda")

    lr = args.learning_rate
    optimizer = flow.optim.RMSprop(net.parameters(), lr, weight_decay=args.weight_decay)

    criterion = nn.BCELoss()
    epoch = args.epochs
    num_steps = len(train_loader)
    for i in range(epoch):

        net.train()
        epoch_loss = 0

        for step, data in enumerate(train_loader):
            images, labels = data
            images = images.permute(0, 3, 1, 2)
            images = images / 255.0
            images = images.to("cuda", dtype=flow.float32)
            labels = labels.to("cuda", dtype=flow.float32)

            pred = net(images)
            loss = criterion(pred, labels)
            epoch_loss += loss.numpy()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            lr = optimizer.param_groups[0]["lr"]
            print("Train:[%d/%d][%d/%d] Training Loss: %.4f Lr: %.6f" % (
                (i + 1),
                args.epochs,
                step,
                num_steps,
                loss.numpy(),
                lr
            ))
        filename = "UNetmodel_Epoch_" + str(i)
        save_checkpoint_path = args.save_checkpoint_path
        flow.save(net.state_dict(), os.path.join(save_checkpoint_path ,filename))
        print("save net successfully!")

if __name__ == "__main__":
    args = _parse_args()
    main(args)