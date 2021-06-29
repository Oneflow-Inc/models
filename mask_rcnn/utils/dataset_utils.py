import copy
import os
from PIL import Image

import oneflow.experimental as flow
import oneflow.experimental.nn as nn

from pycocotools import mask as coco_mask
from pycocotools.coco import COCO

from utils.ofrecord_data_utils import OFRecordDataLoader
# import transforms as T


def get_ofrecord(root, image_set, batch_size, transforms, mode='instances'):
    if image_set == "train":
        dataset = OFRecordDataLoader(ofrecord_root = mode, mode = image_set, dataset_size = 9469, batch_size = batch_size)
    elif image_set == "val":
        dataset = OFRecordDataLoader(ofrecord_root= mode, mode=image_set, dataset_size=3925, batch_size=batch_size)
    return dataset



def get_coco(root, image_set, transforms, mode='instances'):
    pass
    # anno_file_template = "{}_{}2017.json"
    # PATHS = {
    #     "train": ("train2017", os.path.join("annotations", anno_file_template.format(mode, "train"))),
    #     "val": ("val2017", os.path.join("annotations", anno_file_template.format(mode, "val"))),
    #     # "train": ("val2017", os.path.join("annotations", anno_file_template.format(mode, "val")))
    # }
    #
    # t = [ConvertCocoPolysToMask()]
    #
    # if transforms is not None:
    #     t.append(transforms)
    # transforms = T.Compose(t)
    #
    # img_folder, ann_file = PATHS[image_set]
    # img_folder = os.path.join(root, img_folder)
    # ann_file = os.path.join(root, ann_file)
    #
    # dataset = CocoDetection(img_folder, ann_file, transforms=transforms)
    #
    # if image_set == "train":
    #     dataset = _coco_remove_images_without_annotations(dataset)
    #
    # # dataset = torch.utils.data.Subset(dataset, [i for i in range(500)])
    #
    # return dataset