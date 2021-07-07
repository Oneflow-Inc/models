from os import listdir
import os
import numpy as np
from PIL import Image, ImageOps


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG'])


def calculate_valid_crop_size(crop_size, upscale_factor):
    return crop_size - (crop_size % upscale_factor)


def load_image(image_path):
    im = Image.open(image_path)
    im = im.convert('RGB')
    w, h = im.size
    im = np.array(im).astype('float32') / 255.
    im = np.transpose(im, (2, 0, 1))
    im = np.expand_dims(im, axis=0)
    return np.ascontiguousarray(im, 'float32'), h, w


class NumpyDataLoader(object):
    def __init__(self, dataset_root: str, mode, hr_size, lr_size, batch_size: int = 1):
        self.dataset_root = dataset_root
        """
                image transform: randomly crop, mirror, normalization(0,1), transpose(bs, img_channel, h, w) and shuffle
            """
        self.mode = mode
        self.hr_size = hr_size
        self.lr_size = lr_size
        self.batch_size = batch_size
        self.image_list = []
        self.curr_idx = 0
        images_dir = os.path.join(self.dataset_root, self.mode)
        image_names = os.listdir(images_dir)
        for name in image_names:
            self.image_list.append(os.path.join(images_dir, name))

    def __getitem__(self, index):
        hr_batch, lr_batch = [], []

        for i in range(self.batch_size):
            image_path = self.image_list[index]
            if not is_image_file(image_path):
                print("The file is not an image in:{}, so we continune next one.".format(image_path))
                continue
            img = Image.open(image_path)
            img = img.convert('RGB')

            # random crop crop_size
            w, h = img.size
            if (h > self.hr_size) and (w > self.hr_size):
                x1 = np.random.randint(0, w - self.hr_size)
                y1 = np.random.randint(0, h - self.hr_size)
                hr_img = img.crop((x1, y1, x1 + self.hr_size, y1 + self.hr_size))
                lr_img = hr_img.resize((self.lr_size, self.lr_size))
                if np.random.rand() > 0.5:
                    # random mirroring
                    hr_img = ImageOps.mirror(hr_img)
                    lr_img = ImageOps.mirror(lr_img)

                # normalizing the images to [0, 1]
                hr_img = np.array(hr_img) / 255.
                hr_img = hr_img.astype('float32')
                lr_img = np.array(lr_img) / 255.
                lr_img = lr_img.astype('float32')
                hr_img = hr_img.transpose(2, 0, 1)
                lr_img = lr_img.transpose(2, 0, 1)
                assert hr_img.shape == (3, self.hr_size, self.hr_size), hr_img.shape
                assert lr_img.shape == (3, self.lr_size, self.lr_size), lr_img.shape
                hr_img = np.expand_dims(hr_img, axis=0)
                lr_img = np.expand_dims(lr_img, axis=0)
                hr_batch.append(hr_img)
                lr_batch.append(lr_img)
                self.curr_idx += 1
        if (len(hr_batch) == self.batch_size):
            global hr_datas
            hr_datas = np.concatenate(tuple(hr_batch), axis=0)
            global lr_datas
            lr_datas = np.concatenate(tuple(lr_batch), axis=0)

        return np.ascontiguousarray(hr_datas, 'float32'), np.ascontiguousarray(lr_datas, 'float32')

    def __len__(self):
        return len(self.image_list) // self.batch_size


class ValDatasetFromFolder(object):
    def __init__(self, dataset_dir, mode, upscale_factor):
        self.dataset_dir = dataset_dir
        self.mode = mode
        self.upscale_factor = upscale_factor
        images_dir = os.path.join(self.dataset_dir, self.mode)
        self.image_filenames = [os.path.join(images_dir, x) for x in listdir(images_dir) if is_image_file(x)]

    def __getitem__(self, index):
        hr_image = Image.open(self.image_filenames[index]).convert('RGB')
        w, h = hr_image.size
        crop_size = calculate_valid_crop_size(min(w, h), self.upscale_factor)
        lr_size = crop_size // self.upscale_factor
        hr_image = hr_image.crop((0, 0, crop_size, crop_size))
        lr_image = hr_image.resize((lr_size, lr_size), Image.BICUBIC)

        hr_img = np.array(hr_image) / 255.
        hr_img = hr_img.astype('float32')
        lr_img = np.array(lr_image) / 255.
        lr_img = lr_img.astype('float32')
        hr_img = hr_img.transpose(2, 0, 1)
        lr_img = lr_img.transpose(2, 0, 1)
        hr_img = np.expand_dims(hr_img, axis=0)
        lr_img = np.expand_dims(lr_img, axis=0)

        return np.ascontiguousarray(hr_img, 'float32'), np.ascontiguousarray(lr_img, 'float32')

    def __len__(self):
        return len(self.image_filenames)