import copy
import os
from typing import Union, List

from PIL import Image

import oneflow as flow
import oneflow.nn as nn

from pycocotools import mask as coco_mask
from pycocotools.coco import COCO

# from utils.ofrecord_data_utils import OFRecordDataLoader
import oneflow.utils.vision.transforms.transforms as T


# import transforms as T


# def get_ofrecord(root, image_set, batch_size, transforms, mode='instances'):
#     if image_set == "train":
#         dataset = OFRecordDataLoader(ofrecord_root=mode, mode=image_set, dataset_size=9469, batch_size=batch_size)
#     elif image_set == "val":
#         dataset = OFRecordDataLoader(ofrecord_root=mode, mode=image_set, dataset_size=3925, batch_size=batch_size)
#     return dataset


def convert_coco_poly_to_mask(segmentations, height, width):
    masks = []
    for polygons in segmentations:
        rles = coco_mask.frPyObjects(polygons, height, width)
        mask = coco_mask.decode(rles)
        if len(mask.shape) < 3:
            mask = mask[..., None]
        mask = flow.tensor(mask, dtype=flow.uint8)
        mask = mask.any(dim=2)
        masks.append(mask)
    if masks:
        masks = flow.stack(masks, dim=0)
    else:
        masks = flow.zeros((0, height, width), dtype=flow.uint8)
    return masks


class ConvertCocoPolysToMask(object):
    def __call__(self, image, target):
        w, h = image.size

        image_id = target["image_id"]
        image_id = flow.tensor([image_id])

        anno = target["annotations"]

        anno = [obj for obj in anno if obj['iscrowd'] == 0]

        boxes = [obj["bbox"] for obj in anno]
        # guard against no boxes via resizing
        boxes = flow.tensor(boxes, dtype=flow.float32).reshape(-1, 4)
        boxes[:, 2:] += boxes[:, :2]
        boxes[:, 0::2].clamp_(min=0, max=w)
        boxes[:, 1::2].clamp_(min=0, max=h)

        classes = [obj["category_id"] for obj in anno]
        classes = flow.tensor(classes, dtype=flow.int64)

        segmentations = [obj["segmentation"] for obj in anno]
        masks = convert_coco_poly_to_mask(segmentations, h, w)

        keypoints = None
        if anno and "keypoints" in anno[0]:
            keypoints = [obj["keypoints"] for obj in anno]
            keypoints = flow.tensor(keypoints, dtype=flow.float32)
            num_keypoints = keypoints.shape[0]
            if num_keypoints:
                keypoints = keypoints.view(num_keypoints, -1, 3)

        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
        boxes = boxes[keep]
        classes = classes[keep]
        masks = masks[keep]
        if keypoints is not None:
            keypoints = keypoints[keep]

        target = {}
        target["boxes"] = boxes
        target["labels"] = classes
        target["masks"] = masks
        target["image_id"] = image_id
        if keypoints is not None:
            target["keypoints"] = keypoints

        # for conversion to coco api
        area = flow.tensor([obj["area"] for obj in anno])
        iscrowd = flow.tensor([obj["iscrowd"] for obj in anno])
        target["area"] = area
        target["iscrowd"] = iscrowd

        return image, target


def _coco_remove_images_without_annotations(dataset, cat_list=None):
    def _has_valid_annotation(anno):
        # if it's empty, there is no annotation
        if len(anno) == 0:
            return False
        # if more than 1k pixels occupied in the image
        return sum(obj["area"] for obj in anno) > 1000

    assert isinstance(dataset, torchvision.datasets.CocoDetection)
    ids = []
    for ds_idx, img_id in enumerate(dataset.ids):
        ann_ids = dataset.coco.getAnnIds(imgIds=img_id, iscrowd=None)
        anno = dataset.coco.loadAnns(ann_ids)
        if cat_list:
            anno = [obj for obj in anno if obj["category_id"] in cat_list]
        if _has_valid_annotation(anno):
            ids.append(ds_idx)

    dataset = torch.utils.data.Subset(dataset, ids)
    return dataset


def get_coco(root, image_set, transforms, mode='instances'):
    anno_file_template = "{}_{}2017.json"
    PATHS = {
        "train": ("train2017", os.path.join("annotations", anno_file_template.format(mode, "train"))),
        "val": ("val2017", os.path.join("annotations", anno_file_template.format(mode, "val"))),
    }

    t = [ConvertCocoPolysToMask()]

    if transforms is not None:
        t.append(transforms)
    transforms = T.Compose(t)

    img_folder, ann_file = PATHS[image_set]
    img_folder = os.path.join(root, img_folder)
    ann_file = os.path.join(root, ann_file)

    dataset = CocoDetection(img_folder, ann_file, transforms=transforms)

    if image_set == "train":
        dataset = _coco_remove_images_without_annotations(dataset)

    # dataset = torch.utils.data.Subset(dataset, [i for i in range(500)])

    return dataset


class COCODataLoader(flow.nn.Module):
    def __init__(
            self,
            anno_file="/dataset/mscoco_2017/annotations/instances_val2017.json",
            image_dir="/dataset/mscoco_2017/val2017",
            batch_size=2,
            device=None,
            placement=None,
            sbp=None,
    ):
        super().__init__()
        self.coco_reader = flow.nn.COCOReader(
            annotation_file=anno_file,
            image_dir=image_dir,
            batch_size=batch_size,
            shuffle=True,
            random_seed=12345,
            stride_partition=True,
            device=device,
            placement=placement,
            sbp=sbp,
        )
        self.image_decoder = flow.nn.image.decode(dtype=flow.float32)
        self.resize = flow.nn.image.Resize(target_size=[224, 224], dtype=flow.float32)
        self.dataset_size = len([f for f in os.listdir(image_dir)
                                 if f.endswith('.jpg') and os.path.isfile(os.path.join(image_dir, f))])
        self.coco = COCO(anno_file)

    def get_coco_api(self, image_ids, image_sizes):
        # annotation IDs need to start at 1, not 0, see torchvision issue #1530
        ann_id = 1
        # dataset = {'images': [], 'categories': [], 'annotations': []}
        categories = set()
        target_list = []
        for image_id, image_size in zip(image_ids, image_sizes):
            # find better way to get target
            # targets = ds.get_annotations(img_idx)
            # img, targets = ds[img_idx]
            # image_id = targets["image_id"].item()
            targets = self.coco.loadAnns(self.coco.getAnnIds(imgIds=image_id.numpy().item()))
            img_dict = {}
            img_dict['image_id'] = image_id.numpy().item()
            # img_dict['height'] = image_size[-2].numpy().item()
            # img_dict['width'] = image_size[-1].numpy().item()
            img_dict['boxes'] = []
            for target in targets:
                x, y , w, h = target["bbox"]
                tmp_target = [x - w/2, x + w/2,  y - h/2, y + h/2]
                # tmp_target[:2], tmp_target[2:] = tmp_target[2:], tmp_target[:2]
                img_dict['boxes'].append(tmp_target)
            img_dict['boxes'] = flow.tensor(img_dict['boxes'])
            # bboxes[:, 2:] -= bboxes[:, :2]
            # bboxes = bboxes.tolist()
            img_dict['labels'] = flow.tensor([target['category_id'] for target in targets])
            img_dict['areas'] = (img_dict['boxes'][:, 3] - img_dict['boxes'][:, 1]) * (img_dict['boxes'][:, 2] - img_dict['boxes'][:, 0])
            # areas = targets['area'].tolist()
            img_dict['iscrowd'] = flow.tensor([target['iscrowd'] for target in targets])
            if 'masks' in targets:
                img_dict['masks'] = targets['masks']
                # make masks Fortran contiguous for coco_mask
                masks = masks.permute(0, 2, 1).contiguous().permute(0, 2, 1)
            if 'keypoints' in targets:
                keypoints = targets['keypoints']
                img_dict['keypoints'] = keypoints.reshape(keypoints.shape[0], -1).tolist()
            target_list.append(img_dict)


            # num_objs = bboxes.shape[0]
        #     for i in range(num_objs):
        #         ann = {}
        #         ann['image_id'] = image_id.numpy().item()
        #         ann['bbox'] = bboxes[i]
        #         ann['category_id'] = labels[i]
        #         categories.add(labels[i])
        #         ann['area'] = areas[i]
        #         ann['iscrowd'] = iscrowd[i]
        #         ann['id'] = ann_id
        #         if 'masks' in targets:
        #             ann["segmentation"] = coco_mask.encode(masks[i].numpy())
        #         if 'keypoints' in targets:
        #             ann['keypoints'] = keypoints[i]
        #             ann['num_keypoints'] = sum(k != 0 for k in keypoints[i][2::3])
        #         dataset['annotations'].append(ann)
        #         ann_id += 1
        # dataset['categories'] = [{'id': i} for i in sorted(categories)]
        return target_list
        # coco_ds.dataset = dataset
        # coco_ds.createIndex()
        # return coco_ds

    def forward(self):
        outputs = self.coco_reader()
        # decode images
        image = self.image_decoder(outputs[0])
        fixed_image = self.resize(image)[0]
        fixed_image = fixed_image.permute(0, 3, 1, 2)
        image_id = outputs[1]
        image_size = outputs[2]
        return fixed_image, self.get_coco_api(image_id, image_size)


def get_coco_loader(data_path: str, mode: str, batch_size: int, device: Union[flow.device, str] = None,
                    placement: flow.placement = None, sbp: Union[flow.sbp.sbp, List[flow.sbp.sbp]] = None):
    anno_file = os.path.join(data_path, "annotations", "instances_val2017.json")
    print(anno_file)
    if mode == "train":
        sub_dir = "train2017"
    elif mode == "val":
        sub_dir = "val2017"
    else:
        raise ValueError("mode can only be train or val, but got {}".format(mode))
    image_dir = os.path.join(data_path, sub_dir)
    # coco_reader = flow.nn.COCOReader(
    #     annotation_file=anno_file,
    #     image_dir=image_dir,
    #     batch_size=batch_size,
    #     shuffle=True,
    #     random_seed=12345,
    #     stride_partition=True,
    #     device=device,
    #     placement=placement,
    #     sbp=sbp,
    # )
    coco_loader = COCODataLoader(
        anno_file=anno_file,
        image_dir=image_dir,
        batch_size=batch_size,
        # placement=flow.placement("cpu", {0: [0, 1]}),
        # sbp=[flow.sbp.split(0)],
    )
    num_classes = 91

    return coco_loader, num_classes
