import os

import cv2
import oneflow as flow
import albumentations as albu
import oneflow.utils.data as data


def get_datadir_path(args, split="train"):
    assert split in ["train", "val", "test"]
    if split == "train":
        x_dir = os.path.join(args.data_dir, "train")
        y_dir = os.path.join(args.data_dir, "train_labels")
    elif split == "val":
        x_dir = os.path.join(args.data_dir, "val")
        y_dir = os.path.join(args.data_dir, "valid_labels")
    elif split == "test":
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
