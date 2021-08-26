import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import cv2
import numpy as np

import oneflow as flow
import oneflow.nn as nn
import albumentations as albu
import oneflow.utils.data as data
from oneflow._oneflow_internal import float32


# 设置数据集路径
DATA_DIR = '.../data/CamVid'  # 根据自己的路径来设置

x_train_dir = os.path.join(DATA_DIR, 'train')
y_train_dir = os.path.join(DATA_DIR, 'train_labels')

x_valid_dir = os.path.join(DATA_DIR, 'valid')
y_valid_dir = os.path.join(DATA_DIR, 'valid_labels')

x_test_dir = os.path.join(DATA_DIR, 'test')
y_test_dir = os.path.join(DATA_DIR, 'test_labels')


# 定义Dataloader
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
            self,
            images_dir,
            masks_dir,
            augmentation=None,
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

        # 　抱歉代码写的这么粗暴，意思就是讲mask里的道路设置为前景，而其他设置为背景
        # road
        mask = (mask == 17)
        mask = mask.astype('float')

        # apply augmentations
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        # 这里必须设置一个mask的shape，因为前边的形状是（320,320）
        return image, mask.reshape(1, 320, 320)

    def __len__(self):
        return len(self.ids)



def get_training_augmentation():
    train_transform = [
        albu.HorizontalFlip(p=0.5),
        albu.Resize(height=320, width=320, always_apply=True),
        albu.ShiftScaleRotate(scale_limit=0.1, rotate_limit=20, shift_limit=0.1, p=1, border_mode=0),
    ]
    return albu.Compose(train_transform)


def get_test_augmentation():
    train_transform = [
        albu.Resize(height=320, width=320, always_apply=True),
    ]
    return albu.Compose(train_transform)


augmented_dataset = Dataset(
    x_train_dir,
    y_train_dir,
    augmentation=get_training_augmentation(),
)


# 定义UNet的基本模块
# 代码来自https://github.com/milesial/Pytorch-UNet
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
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)

        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        constantpad= nn.ConstantPad2d([diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x1 = constantpad(x1)

        x = flow.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


# UNet
class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        # 考虑到我电脑的显卡大小，我降低了参数~~，无奈之举
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
        self.out = flow.sigmoid  # 此处记得有sigmoid

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


# 设置train数据集
train_dataset = Dataset(
    x_train_dir,
    y_train_dir,
    augmentation=get_training_augmentation(),
)
batch_size=8
train_loader = data.DataLoader(train_dataset, batch_size, shuffle=True)



# 准备训练，定义模型，只做了两分类
net = UNet(n_channels=3, n_classes=1)
net.to('cuda:0')

# 这里说一下是怎么train的
# 先lr=0.01,train大概40个epoch
# 然后lr=0.005,train大概40个epoch
# 最后在lr=0.0001,train大概20个epoch
lr=0.001
optimizer = flow.optim.RMSprop(net.parameters(), lr, weight_decay=1e-8)

criterion = nn.BCELoss()
epoch=50

for i in range(epoch):

    net.train()
    epoch_loss = 0

    for data in train_loader:
        # 修改一下数据格式
        images, labels = data
        images = images.permute(0, 3, 1, 2)  # 交换通道顺序
        images = images / 255.  # 把image的值归一化到[0,1]
        images = images.to("cuda", dtype=float32)
        labels = labels.to("cuda", dtype=float32)

        pred = net(images)
        loss = criterion(pred, labels)
        epoch_loss += loss.numpy()[0]

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(
            'epoch', i,
            'loss: ', loss.numpy()[0]
        )
    filename="UNetmodel-"+str(i)
    save_checkpoint_path=".../result/"
    flow.save(net.state_dict(), save_checkpoint_path+filename)
    print("save net successfully!")



