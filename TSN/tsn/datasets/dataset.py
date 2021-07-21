from PIL import Image
import os
import os.path
import numpy as np
from numpy.random import randint
from .transform import *

class VideoRecord(object):
    def __init__(self, row):
        self._data = row

    @property
    def path(self):
        return self._data[0]

    @property
    def num_frames(self):
        return int(self._data[1])

    @property
    def label(self):
        return int(self._data[2])


class TSNDataSet(object):
    def __init__(self, root_path, list_file, video_dir='/home/liling/work/oneflow/mmaction/data/kinetics400/rawframes_val',
                 num_segments=3, new_length=1, modality='RGB',
                 image_tmpl='img_{:05d}.jpg', transform=None,
                 flip = None, crop = None, sample = None, stack=None, totensor=None,
                 Normalize=None, batch_size=8, force_grayscale=False, random_shift=True, test_mode=False):

        self.root_path = root_path
        self.list_file = list_file
        self.num_segments = num_segments
        self.new_length = new_length
        self.modality = modality
        self.image_tmpl = image_tmpl
        self.transform = transform
        self.flip = flip
        self.crop = crop
        self.sample = sample
        self.stack = stack
        self.Normalize = Normalize
        self.batch_size = batch_size
        self.random_shift = random_shift
        self.test_mode = test_mode
        self.data_dir = video_dir

        if self.modality == 'RGBDiff':
            self.new_length += 1      # Diff needs one more image to calculate diff

        self._parse_list()

        self.curr_idx = 0
        if test_mode:
            self.shuffle_data()

    def shuffle_data(self):
        random.shuffle(self.video_list)
        self.curr_idx = 0

    def _load_image(self, directory, idx):
        if self.modality == 'RGB' or self.modality == 'RGBDiff':
            return [Image.open(os.path.join(directory, self.image_tmpl.format(idx))).convert('RGB')]
        elif self.modality == 'Flow':
            x_img = Image.open(os.path.join(directory, self.image_tmpl.format('x', idx))).convert('L')
            y_img = Image.open(os.path.join(directory, self.image_tmpl.format('y', idx))).convert('L')

            return [x_img, y_img]

    def _parse_list(self):
        self.video_list = [VideoRecord(x.strip().split(' ')) for x in open(self.list_file)]

    def _sample_indices(self, record):
        """
        :param record: VideoRecord
        :return: list
        """

        average_duration = (record.num_frames - self.new_length + 1) // self.num_segments
        if average_duration > 0:
            offsets = np.multiply(list(range(self.num_segments)), average_duration) + randint(average_duration, size=self.num_segments)
        elif record.num_frames > self.num_segments:
            offsets = np.sort(randint(record.num_frames - self.new_length + 1, size=self.num_segments))
        else:
            offsets = np.zeros((self.num_segments,))
        return offsets + 1

    def _get_val_indices(self, record):
        if record.num_frames > self.num_segments + self.new_length - 1:
            tick = (record.num_frames - self.new_length + 1) / float(self.num_segments)
            offsets = np.array([int(tick / 2.0 + tick * x) for x in range(self.num_segments)])
        else:
            offsets = np.zeros((self.num_segments,))
        return offsets + 1

    def _get_test_indices(self, record):

        tick = (record.num_frames - self.new_length + 1) / float(self.num_segments)

        offsets = np.array([int(tick / 2.0 + tick * x) for x in range(self.num_segments)])

        return offsets + 1

    def get_ann_info(self, idx):
        return {
            'path': self.video_list[idx].path,
            'label': self.video_list[idx].label
        }

    def __getitem__(self, index):
        batch_datas=[]
        batch_labels=[]
        if index == 0:
            self.curr_idx = 0

        for i in range(self.batch_size):
            record = self.video_list[self.curr_idx]

            if not self.test_mode:
                segment_indices = self._sample_indices(record) if self.random_shift else self._get_val_indices(record)
            else:
                segment_indices = self._get_test_indices(record)

            data, label = self.get(record, segment_indices)
            batch_datas.append(data)
            batch_labels.append(label)
            self.curr_idx += 1

        np_datas = np.concatenate(tuple(batch_datas), axis=0)
        np_labels = np.array(batch_labels, dtype=np.int32)

        return np.ascontiguousarray(np_datas, 'float32'), np_labels

    def get(self, record, indices):
        images = list()
        new_path = self.data_dir + '/' + record.path

        for seg_ind in indices:
            p = int(seg_ind)
            for i in range(self.new_length):
                seg_imgs = self._load_image(new_path, p)
                images.extend(seg_imgs)
                if p < record.num_frames:
                    p += 1

        if self.test_mode:
            s_images = self.sample(images)
        else:
            c_images = self.crop(images)
            s_images = self.flip(c_images)

        ss_images = self.stack(s_images)
        ss_tensor = ToFlowFormatTensor(ss_images)
        process_data = self.Normalize(ss_tensor)

        return process_data, record.label

    def __len__(self):
        return len(self.video_list)
