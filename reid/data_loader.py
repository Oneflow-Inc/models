"""
Copyright 2020 The OneFlow Authors. All rights reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

#-*- coding:utf-8 -*-
# Version: 0.0.1
# Author: scorpio.lu(luyi@zhejianglab.com)
# Data: 06/28/2020

import os
import os.path as osp
from glob import glob
import re
import errno
import sys
import time
from six.moves import urllib
import tarfile
import zipfile
from collections import defaultdict
import copy, random
import numpy as np
from PIL import Image, ImageOps
import math

class Market1501(object):
    """
    Market1501
    Reference:
    Zheng et al. Scalable Person Re-identification: A Benchmark. ICCV 2015.
    URL: http://www.liangzheng.org/Project/project_reid.html

    Dataset statistics:
    # identities: 1501 (+1 for background)
    # images: 12936 (train) + 3368 (query) + 15913 (gallery)
    """
    dataset_dir = 'market1501'
    dataset_url = 'http://188.138.127.15:81/Datasets/Market-1501-v15.09.15.zip'

    def __init__(self, root='/home/data', show_summery=True):
        super(Market1501, self).__init__()
        self.dataset_dir = os.path.join(root, self.dataset_dir)

        self.train_dir = os.path.join(self.dataset_dir, 'bounding_box_train')
        self.query_dir = os.path.join(self.dataset_dir, 'query')
        self.gallery_dir = os.path.join(self.dataset_dir, 'bounding_box_test')
        self.download_dataset(self.dataset_dir, self.dataset_url)
        # check files
        self._check_before_run()
        # process train(need relabel), query, gallery
        self.train = self._process(self.train_dir, relabel=True)
        self.query = self._process(self.query_dir, relabel=False)
        self.gallery = self._process(self.gallery_dir, relabel=False)
        # dataset statistics
        self.num_train_pids, self.num_train_imgs, self.num_train_camids = self.get_dataset_info(self.train)
        self.num_query_pids, self.num_query_imgs, self.num_query_camids = self.get_dataset_info(self.query)
        self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_camids = self.get_dataset_info(self.gallery)

        if show_summery:
            print("=> Market1501 has loaded")
            self._print_dataset_info()

    def get_dataset_info(self, data):
        """
           get dataset information: num of images, pids, camera ids
           Args:
               data : dataset.
        """
        pids, cams = [], []
        for _, pid, camid in data:
            pids += [pid]
            cams += [camid]
        pids = set(pids)
        cams = set(cams)
        num_pids = len(pids)
        num_cams = len(cams)
        num_imgs = len(data)
        return num_pids, num_imgs, num_cams

    def _check_before_run(self):
        """Check if all files are available"""
        if not os.path.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))
        if not os.path.exists(self.train_dir):
            raise RuntimeError("'{}' is not available".format(self.train_dir))
        if not os.path.exists(self.query_dir):
            raise RuntimeError("'{}' is not available".format(self.query_dir))
        if not os.path.exists(self.gallery_dir):
            raise RuntimeError("'{}' is not available".format(self.gallery_dir))

    def _print_dataset_info(self):
        print("Dataset information:")
        print("  ----------------------------------------")
        print("  subset   | # ids | # images | # cameras")
        print("  ----------------------------------------")
        print("  train    | {:5d} | {:8d} | {:9d}".format(self.num_train_pids, self.num_train_imgs, self.num_train_camids))
        print("  query    | {:5d} | {:8d} | {:9d}".format(self.num_query_pids, self.num_query_imgs, self.num_query_camids))
        print("  gallery  | {:5d} | {:8d} | {:9d}".format(self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_camids))
        print("  ----------------------------------------")

    def _process(self, dir_path, relabel=False):
        """construct dataset"""
        img_paths = glob(os.path.join(dir_path, '*.jpg'))
        pattern = re.compile(r'([-\d]+)_c(\d)')
        pid_container = set()
        for img_path in img_paths:
            pid, _ = map(int, pattern.search(img_path).groups())
            if pid == -1:
                continue  # junk images are just ignored
            pid_container.add(pid)
        pid2label = {pid: label for label, pid in enumerate(pid_container)}

        dataset = []
        for img_path in img_paths:
            pid, camid = map(int, pattern.search(img_path).groups())
            if pid == -1:
                continue  # junk images are just ignored
            assert 0 <= pid <= 1501  # pid == 0 means background
            assert 1 <= camid <= 6
            camid -= 1  # index starts from 0
            if relabel:
                pid = pid2label[pid]
            dataset.append([img_path, pid, camid])

        return dataset

    def download_dataset(self, dir, url):
        """Download and extract the corresponding dataset to the 'dataset_dir' .
        Args:
            dir (str): dataset directory.
            url (str): url where to download dataset.
        """
        if os.path.exists(dir):
            return

        if url is None:
            raise RuntimeError(
                '{} dataset needs to be manually '
                'prepared, please follow the '
                'document to prepare this dataset'.format(
                    self.__class__.__name__
                )
            )

        print('Creating directory "{}"'.format(dir))
        if not os.path.exists(dir):
            try:
                os.makedirs(dir)
            except OSError as e:
                if e.errno != errno.EEXIST:
                    raise

        fpath = os.path.join(dir, os.path.basename(url))
        # download dataset
        print(
            'Downloading {} dataset to "{}"'.format(
                self.__class__.__name__, dir
            )
        )
        self.download_url(url, fpath)
        # extract dataset to the destination
        print('Extracting "{}"'.format(fpath))
        try:
            tar = tarfile.open(fpath)
            tar.extractall(path=dir)
            tar.close()
        except:
            zip_ref = zipfile.ZipFile(fpath, 'r')
            zip_ref.extractall(dir)
            zip_ref.close()

        print('{} dataset is ready'.format(self.__class__.__name__))

    def download_url(self, url, dst):
        """Downloads file from a url to a destination.
        Args:
            url (str): url to download file.
            dst (str): destination path.
        """

        print('* url="{}"'.format(url))
        print('* destination="{}"'.format(dst))

        def _reporthook(count, block_size, total_size):
            global start_time
            if count == 0:
                start_time = time.time()
                return
            duration = time.time() - start_time
            progress_size = int(count * block_size)
            speed = int(progress_size / (1024 * duration))
            percent = int(count * block_size * 100 / total_size)
            sys.stdout.write(
                '\r...%d%%, %d MB, %d KB/s, %d seconds passed' %
                (percent, progress_size / (1024 * 1024), speed, duration)
            )
            sys.stdout.flush()

        urllib.request.urlretrieve(url, dst, _reporthook)
        sys.stdout.write('\n')

class RandomIdentitySampler():
    """Randomly samples N identities each with K instances.

    Args:
        data_source (list): contains tuples of (img_path(s), pid, camid).
        batch_size (int): batch size.
        num_instances (int): number of instances per identity in a batch.
    """
    def __init__(self, data_source, batch_size, num_instances):
        if batch_size < num_instances:
            raise ValueError('batch_size={} must be no less '
                             'than num_instances={}'.format(batch_size, num_instances))

        self.data_source = data_source
        self.batch_size = batch_size
        self.num_instances = num_instances
        self.num_pids_per_batch = self.batch_size // self.num_instances
        self.index_dic = defaultdict(list)
        for index, pid in enumerate(self.data_source):
            self.index_dic[pid].append(index)
        self.pids = list(self.index_dic.keys())

        # estimate number of examples in an epoch
        # TODO: improve precision
        self.length = 0
        for pid in self.pids:
            idxs = self.index_dic[pid]
            num = len(idxs)
            if num < self.num_instances:
                num = self.num_instances
            self.length += num - num % self.num_instances

    def __iter__(self):
        batch_idxs_dict = defaultdict(list)

        for pid in self.pids:
            idxs = copy.deepcopy(self.index_dic[pid])
            if len(idxs) < self.num_instances:
                idxs = np.random.choice(idxs, size=self.num_instances, replace=True)
            random.shuffle(idxs)
            batch_idxs = []
            for idx in idxs:
                batch_idxs.append(idx)
                if len(batch_idxs) == self.num_instances:
                    batch_idxs_dict[pid].append(batch_idxs)
                    batch_idxs = []

        avai_pids = copy.deepcopy(self.pids)
        final_idxs = []

        while len(avai_pids) >= self.num_pids_per_batch:
            selected_pids = random.sample(avai_pids, self.num_pids_per_batch)
            for pid in selected_pids:
                batch_idxs = batch_idxs_dict[pid].pop(0)
                final_idxs.extend(batch_idxs)
                if len(batch_idxs_dict[pid]) == 0:
                    avai_pids.remove(pid)

        return iter(final_idxs)

    def __len__(self):
        return self.length


class RandomErasing(object):
    """Randomly erases an image patch.

    Origin: `<https://github.com/zhunzhong07/Random-Erasing>`_

    Reference:
        Zhong et al. Random Erasing Data Augmentation.

    Args:
        probability (float, optional): probability that this operation takes place.
            Default is 0.5.
        sl (float, optional): min erasing area.
        sh (float, optional): max erasing area.
        r1 (float, optional): min aspect ratio.
        mean (list, optional): erasing value.
    """

    def __init__(self, probability=0.5, sl=0.02, sh=0.4, r1=0.3, mean=[0.4914, 0.4822, 0.4465]):
        self.probability = probability
        self.mean = mean
        self.sl = sl
        self.sh = sh
        self.r1 = r1

    def __call__(self, img):
        if random.uniform(0, 1) > self.probability:
            return img

        for attempt in range(100):
            area = img.shape[1] * img.shape[2]

            target_area = random.uniform(self.sl, self.sh) * area
            aspect_ratio = random.uniform(self.r1, 1 / self.r1)

            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))

            if w < img.shape[2] and h < img.shape[1]:
                x1 = random.randint(0, img.shape[1] - h)
                y1 = random.randint(0, img.shape[2] - w)
                if img.shape[0] == 3:
                    img[0, x1:x1 + h, y1:y1 + w] = self.mean[0]
                    img[1, x1:x1 + h, y1:y1 + w] = self.mean[1]
                    img[2, x1:x1 + h, y1:y1 + w] = self.mean[2]
                else:
                    img[0, x1:x1 + h, y1:y1 + w] = self.mean[0]
                return img

        return img

def resize(img, size, interpolation=Image.BILINEAR):
    r"""Resize the input PIL Image to the given size.

    Args:
        img (PIL Image): Image to be resized.
        size (sequence or int): Desired output size. If size is a sequence like
            (h, w), the output size will be matched to this. If size is an int,
            the smaller edge of the image will be matched to this number maintaing
            the aspect ratio. i.e, if height > width, then image will be rescaled to
            :math:`\left(\text{size} \times \frac{\text{height}}{\text{width}}, \text{size}\right)`
        interpolation (int, optional): Desired interpolation. Default is
            ``PIL.Image.BILINEAR``

    Returns:
        PIL Image: Resized image.
    """
    if isinstance(size, int):
        w, h = img.size
        if (w <= h and w == size) or (h <= w and h == size):
            return img
        if w < h:
            ow = size
            oh = int(size * h / w)
            return img.resize((ow, oh), interpolation)
        else:
            oh = size
            ow = int(size * w / h)
            return img.resize((ow, oh), interpolation)
    else:
        return img.resize(size[::-1], interpolation)

class ImageDataset():
    """A base class representing ImageDataset.

    All other image datasets should subclass it.

    ``__getitem__`` returns an image given index.
    It will return ``img``, ``pid``, ``camid`` and ``img_path``
    where ``img`` has shape (channel, height, width). As a results,
    data in each batch has shape (batch_size, channel, height, width).
    """

    def __init__(self, dataset,flag, process_size):
        super(ImageDataset, self).__init__()
        self.dataset = np.array(dataset)
        self.flag = flag
        self.height, self.width = process_size

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img_path, pid, camid = self.dataset[index]
        if self.flag == 'train':
            img = read_and_preprocess_image(img_path,self.width,self.height)
        else:
            img = read_test_image(img_path,self.width, self.height)
        imgs = []
        imgs.append(img)
        imgs = np.asarray(imgs).astype(np.float32)
        return imgs, pid, camid

    def __getbatch__(self, index):
        img_paths, pid, camid = zip(*self.dataset[index])
        pid = list(map(int,pid))
        camid = list(map(int,camid))
        imgs = []
        if self.flag == 'train':
            for img_path in img_paths:
                img = read_and_preprocess_image(img_path,self.width,self.height)
                imgs.append(img)
        else:
            for img_path in img_paths:
                img = read_test_image(img_path,self.width,self.height)
                imgs.append(img)
        imgs = np.asarray(imgs).astype(np.float32)
        return imgs, np.array(pid), np.array(camid)



def read_test_image(path,width,height):
    """Reads image from path using ``PIL.Image``.

    Args:
        path (str): path to an image.

    Returns:
        PIL image
    """
    got_img = False
    rgb_mean = [0.485, 0.456, 0.406]
    rgb_std = [0.229, 0.224, 0.225]
    if not osp.exists(path):
        raise IOError('"{}" does not exist'.format(path))
    while not got_img:
        try:
            img = Image.open(path).convert('RGB')
            img = resize(img, (height,width))
            img = np.array(img).astype(np.float32) / 255.
            img = (img - rgb_mean) / rgb_std
            got_img = True
        except IOError:
            print('IOError incurred when reading "{}". Will redo. Don\'t worry. Just chill.'.format(path))
    return img.transpose(2, 0, 1).astype(np.float32)



def read_and_preprocess_image(path,width,height):
    """Reads image from path using ``PIL.Image``.

    Args:
        path (str): path to an image.

    Returns:
        PIL image
    """
    RandErasing = RandomErasing()
    got_img = False
    rgb_mean = [0.485, 0.456, 0.406]
    rgb_std = [0.229, 0.224, 0.225]
    if not osp.exists(path):
        raise IOError('"{}" does not exist'.format(path))
    while not got_img:
        try:
            img = Image.open(path).convert('RGB')
            img = resize(img, (height, width))
            img = RandomHorizontalFlip(img)
            img = RandomCrop(img,(height, width))

            img = np.array(img).astype(np.float32) / 255.
            img = (img - rgb_mean) / rgb_std
            img = img.transpose(2, 0, 1)
            img = RandErasing(img)
            got_img = True
        except IOError:
            print('IOError incurred when reading "{}". Will redo. Don\'t worry. Just chill.'.format(path))
    return img.astype(np.float32)





def RandomCrop(img, size, padding = 10, fill = 0):
    if padding is not None:
        if img.mode == 'P':
            palette = img.getpalette()
            image = ImageOps.expand(img, border=padding, fill=fill)
            image.putpalette(palette)
            img = image
        img = ImageOps.expand(img, border=padding, fill=fill)
    i, j, h, w = get_params(img, size)

    return img.crop((j, i, j + w, i + h))

def get_params(img, output_size):
    """Get parameters for ``crop`` for a random crop.
    Args:
        img (PIL Image): Image to be cropped.
        output_size (tuple): Expected output size of the crop.
    Returns:
        tuple: params (i, j, h, w) to be passed to ``crop`` for random crop.
    """
    w, h = img.size
    th, tw = output_size
    if w == tw and h == th:
        return 0, 0, h, w

    i = random.randint(0, h - th)
    j = random.randint(0, w - tw)
    return i, j, th, tw

def RandomHorizontalFlip(img):
    if random.random() < 0.5:
        return img.transpose(Image.FLIP_LEFT_RIGHT)
    return img
