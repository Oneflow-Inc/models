import oneflow
from oneflow.utils.data import Dataset
import cv2
import os
import glob
import random


class SelfDataSet(Dataset):
    def __init__(self, data_path):
        self.data_path = data_path
        self.imgs_path = glob.glob(os.path.join(data_path, "*.png"))

    def augment(self, image, flipcode):
        flip = cv2.flip(image, flipcode)
        return flip

    def __getitem__(self, index):
        image_path = self.imgs_path[index]
        label_path = image_path.replace("image", "label")
        image = cv2.imread(image_path)
        label = cv2.imread(label_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        label = cv2.cvtColor(label, cv2.COLOR_BGR2GRAY)
        image = image.reshape(1, image.shape[0], image.shape[1])
        label = label.reshape(1, label.shape[0], label.shape[1])

        if label.max() > 1:
            label = label / 255
        flipcode = random.choice([-1, 0, 1, 2])
        if flipcode != 2:
            image = self.augment(image, flipcode)
            label = self.augment(label, flipcode)
        return image, label

    def __len__(self):
        return len(self.imgs_path)


if __name__ == "__main__":
    data_path = "train_image"
    plate_dataset = SelfDataSet(data_path)
    print(len(plate_dataset))
    train_loader = oneflow.utils.data.DataLoader(
        dataset=plate_dataset, batch_size=5, shuffle=True
    )
    for image, label in train_loader:
        print(label.shape)
