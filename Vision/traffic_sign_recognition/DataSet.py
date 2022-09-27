from oneflow.utils.data import Dataset
from flowvision import transforms as ft
import oneflow as of
import json
import cv2
import numpy as np

def AddGaussianNoise(array, mean=0, std=0.02):

    noise = np.random.normal(mean, std, array.shape)
    array = array / 255 + noise
    array[array < 0] = 0
    array[array > 1] = 1
    return np.uint8(array * 255)


var_list1 = [0.02, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
var_list2 = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1]

class MyDataset1(Dataset):

    def __init__(self, json_path='train_list1.json', if_train=True):
        super(MyDataset1, self).__init__()

        # read json file
        file = open(json_path)
        infos = json.load(file)
        annotations = infos['annotations']

        self.train_list = annotations

        self.len = len(annotations)

        self.if_train = if_train

        self.transforms = ft.Compose([
            ft.ToTensor(),
            ft.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ]
        )

    def __getitem__(self, item):
        file1 = self.train_list[item]  # a dict
        img_file = file1['filename']  # path of image
        class1 = int(file1['label'])  # label of image

        img = cv2.imread(img_file)
        
        if self.if_train:
            t11 = of.randint(0, 21, size=[1]).item()
            t12 = of.randint(1, 21, size=[1]).item()
            t13 = of.randint(1, 21, size=[1]).item()
        else:
            t11 = 20
            t12 = of.randint(1, 21, size=[1]).item()
            t13 = of.randint(1, 21, size=[1]).item()
        
        if t11 <= 18:
            t3 = of.randint(0, 7, size=[1]).item()
            img = AddGaussianNoise(img, std=var_list1[t3])
            img = cv2.blur(img, ksize=(9, 9))
        else:
            img = cv2.blur(img, ksize=(9, 9))
        # rotate
        if t12 <= 2:
            img = cv2.rotate(img, 2)
        elif t12 <= 4:
            img = cv2.rotate(img, 0)
        elif t12 <= 6:
            img = cv2.rotate(img, 1)
        else:
            pass

        # flip
        if t13 <= 2:
            img = cv2.flip(img, 0)
        elif t13 <= 4:
            img = cv2.flip(img, 1)
        elif t13 <= 6:
            img = cv2.flip(img, -1)
        else:
            pass

        img = self.transforms(img)

        return img, of.tensor(class1, dtype=of.int)

    def __len__(self):
        return self.len

class MyDataset2(Dataset):

    def __init__(self, json_path='train_list1.json', if_train=True):
        super(MyDataset2, self).__init__()
        # read json file
        file = open(json_path)
        infos = json.load(file)
        annotations = infos['annotations']

        self.train_list = annotations

        self.len = len(annotations)

        self.if_train = if_train

        self.transforms = ft.Compose([
            ft.ToTensor(),
            ft.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]
        )

    def __getitem__(self, item):
        file1 = self.train_list[item]  # a dict
        img_file = file1['filename']  # path of image
        class1 = int(file1['label'])  # label of image
        img = cv2.imread(img_file)

        if self.if_train:
            t11 = of.randint(0, 12, size=[1]).item()
            t12 = of.randint(1, 21, size=[1]).item()
            t13 = of.randint(1, 21, size=[1]).item()
        else:
            t11 = 4
            t12 = 10
            t13 = 8
        if t11 <= 5:
            t2 = of.randint(0, 10, size=[1])
            img = AddGaussianNoise(img, std=var_list2[t2.item()])
            img = cv2.blur(img, ksize=(9, 9))
        else:
            img = cv2.blur(img, ksize=(9, 9))
        # rotate
        if t12 <= 2:
            img = cv2.rotate(img, 2)
        elif t12 <= 4:
            img = cv2.rotate(img, 0)
        elif t12 <= 6:
            img = cv2.rotate(img, 1)
        else:
            pass
        # flip
        if t13 <= 2:
            img = cv2.flip(img, 0)
        elif t13 <= 4:
            img = cv2.flip(img, 1)
        elif t13 <= 6:
            img = cv2.flip(img, -1)
        else:
            pass

        img = self.transforms(img)

        return img, of.tensor(class1)

    def __len__(self):
        return self.len


if __name__ == '__main__':
    a = MyDataset1()
    b, c = a.__getitem__(1)
    print(b.size())
    print(c)
