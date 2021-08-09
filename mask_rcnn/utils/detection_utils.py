import math

import numpy as np
import oneflow as flow
from oneflow import Tensor

# import torch
# from torch import Tensor
from typing import List, Tuple
from models.frozen_bn import FrozenBatchNorm2d

# from torchvision.ops.misc import FrozenBatchNorm2d


class BalancedPositiveNegativeSampler(object):
    """
    This class samples batches, ensuring that they contain a fixed proportion of positives
    """

    def __init__(self, batch_size_per_image, positive_fraction):
        # type: (int, float) -> None
        """
        Args:
            batch_size_per_image (int): number of elements to be selected per image
            positive_fraction (float): percentace of positive elements per batch
        """
        self.batch_size_per_image = batch_size_per_image
        self.positive_fraction = positive_fraction

    def __call__(self, matched_idxs):
        # type: (List[Tensor]) -> Tuple[List[Tensor], List[Tensor]]
        """
        Args:
            matched idxs: list of tensors containing -1, 0 or positive values.
                Each tensor corresponds to a specific image.
                -1 values are ignored, 0 are considered as negatives and > 0 as
                positives.

        Returns:
            pos_idx (list[tensor])
            neg_idx (list[tensor])

        Returns two lists of binary masks for each image.
        The first list contains the positive elements that were selected,
        and the second list the negative example.
        """
        pos_idx = []
        neg_idx = []
        for matched_idxs_per_image in matched_idxs:
            # positive = torch.where(matched_idxs_per_image >= 1)[0]
            # negative = torch.where(matched_idxs_per_image == 0)[0]
            #TODO:argwhere 0-dim tensor
            positive_list = []
            negative_list = []
            for i in range(matched_idxs_per_image.shape[0]):
                if matched_idxs_per_image[i].numpy().item() >= 1:
                    positive_list.append(i)
                elif matched_idxs_per_image[i].numpy().item() == 0:
                    negative_list.append(i)
            positive = flow.tensor(positive_list, device = matched_idxs_per_image.device, dtype = flow.int64)
            negative = flow.tensor(negative_list, device = matched_idxs_per_image.device, dtype=flow.int64)

            num_pos = int(self.batch_size_per_image * self.positive_fraction)
            # protect against not enough positive examples
            num_pos = min(positive.numel(), num_pos)
            num_neg = self.batch_size_per_image - num_pos
            # protect against not enough negative examples
            num_neg = min(negative.numel(), num_neg)

            # randomly select positive and negative examples
            perm1 = flow.randperm(positive.numel(), device=positive.device)[:num_pos]
            perm2 = flow.randperm(negative.numel(), device=negative.device)[:num_neg]
            # perm1 = np.random.randint(0, positive.numel(), size=1).item()
            pos_idx_per_image = positive[perm1]
            # create binary mask from indices
            pos_idx_per_image_mask = flow.zeros_like(
                matched_idxs_per_image
            )
            for idx in pos_idx_per_image:
                pos_idx_per_image_mask[idx.numpy().item()] = flow.tensor(1).to(device=pos_idx_per_image_mask.device, dtype = pos_idx_per_image_mask.dtype)
            pos_idx.append(pos_idx_per_image_mask)
            neg_idx_per_image = negative[perm2]
            neg_idx_per_image_mask = flow.zeros_like(
                matched_idxs_per_image
            )
            for idx in neg_idx_per_image:
                neg_idx_per_image_mask[idx.numpy().item()] = flow.tensor(1).to(device=pos_idx_per_image_mask.device, dtype = pos_idx_per_image_mask.dtype)
            neg_idx.append(neg_idx_per_image_mask)

        return pos_idx, neg_idx


# @torch.jit._script_if_tracing
def encode_boxes(reference_boxes, proposals, weights):
    # type: (Tensor, Tensor,Tensor) -> Tensor
    """
    Encode a set of proposals with respect to some
    reference boxes

    Args:
        reference_boxes (Tensor): reference boxes
        proposals (Tensor): boxes to be encoded
        weights (Tensor[4]): the weights for ``(x, y, w, h)``
    """

    # perform some unpacking to make it JIT-fusion friendly
    wx = weights[0]
    wy = weights[1]
    ww = weights[2]
    wh = weights[3]

    proposals_x1 = proposals[:, 0].unsqueeze(1)
    proposals_y1 = proposals[:, 1].unsqueeze(1)
    proposals_x2 = proposals[:, 2].unsqueeze(1)
    proposals_y2 = proposals[:, 3].unsqueeze(1)


    reference_boxes_x1 = reference_boxes[:, 0].unsqueeze(1)
    reference_boxes_y1 = reference_boxes[:, 1].unsqueeze(1)
    reference_boxes_x2 = reference_boxes[:, 2].unsqueeze(1)
    reference_boxes_y2 = reference_boxes[:, 3].unsqueeze(1)

    # implementation starts here
    ex_widths = proposals_x2 - proposals_x1
    ex_heights = proposals_y2 - proposals_y1
    ex_ctr_x = proposals_x1 + 0.5 * ex_widths
    ex_ctr_y = proposals_y1 + 0.5 * ex_heights

    gt_widths = reference_boxes_x2 - reference_boxes_x1
    gt_heights = reference_boxes_y2 - reference_boxes_y1
    gt_ctr_x = reference_boxes_x1 + 0.5 * gt_widths
    gt_ctr_y = reference_boxes_y1 + 0.5 * gt_heights

    targets_dx = wx * (gt_ctr_x - ex_ctr_x) / ex_widths
    targets_dy = wy * (gt_ctr_y - ex_ctr_y) / ex_heights
    targets_dw = ww * flow.log(gt_widths / ex_widths)
    targets_dh = wh * flow.log(gt_heights / ex_heights)

    targets = flow.cat((targets_dx, targets_dy, targets_dw, targets_dh), dim=1)
    return targets


class BoxCoder(object):
    """
    This class encodes and decodes a set of bounding boxes into
    the representation used for training the regressors.
    """

    def __init__(self, weights, bbox_xform_clip=math.log(1000. / 16)):
        # type: (Tuple[float, float, float, float], float) -> None
        """
        Args:
            weights (4-element tuple)
            bbox_xform_clip (float)
        """
        self.weights = weights
        self.bbox_xform_clip = bbox_xform_clip

    def encode(self, reference_boxes, proposals):
        # type: (List[Tensor], List[Tensor]) -> List[Tensor]
        boxes_per_image = [b.shape[0] for b in reference_boxes]
        reference_boxes = flow.cat(reference_boxes, dim=0)
        proposals = flow.cat(proposals, dim=0).view(-1, 4)
        # proposals = flow.cat(proposals, dim=0)
        targets = self.encode_single(reference_boxes, proposals)
        #TODO: split op
        split_targets = []
        box_idx = 0
        for box_size in boxes_per_image:
            split_targets.append(targets[box_idx:box_idx + box_size])
            box_idx += box_size
        return split_targets
        # return targets.split(boxes_per_image, 0)


    def encode_single(self, reference_boxes, proposals):
        """
        Encode a set of proposals with respect to some
        reference boxes

        Args:
            reference_boxes (Tensor): reference boxes
            proposals (Tensor): boxes to be encoded
        """
        dtype = reference_boxes.dtype
        device = reference_boxes.device
        weights = flow.Tensor(self.weights, dtype=dtype, device=device)
        targets = encode_boxes(reference_boxes, proposals, weights)

        return targets

    def decode(self, rel_codes, boxes):
        # type: (Tensor, List[Tensor]) -> Tensor
        assert isinstance(boxes, (list, tuple))
        # assert isinstance(rel_codes, Tensor)
        boxes_per_image = [b.size(0) for b in boxes]
        concat_boxes = flow.cat(boxes, dim=0)
        box_sum = 0
        for val in boxes_per_image:
            box_sum += val
        if box_sum > 0:
            rel_codes = rel_codes.reshape((box_sum, -1))
        pred_boxes = self.decode_single(
            rel_codes, concat_boxes
        )
        if box_sum > 0:
            pred_boxes = pred_boxes.reshape((box_sum, -1, 4))
        return pred_boxes

    def decode_single(self, rel_codes, boxes):
        """
        From a set of original boxes and encoded relative box offsets,
        get the decoded boxes.

        Args:
            rel_codes (Tensor): encoded boxes
            boxes (Tensor): reference boxes.
        """

        boxes = boxes.to(rel_codes.dtype)
        widths = boxes[..., 2] - boxes[..., 0]
        heights = boxes[..., 3] - boxes[..., 1]
        ctr_x = boxes[..., 0] + 0.5 * widths
        ctr_y = boxes[..., 1] + 0.5 * heights

        wx, wy, ww, wh = self.weights
        dx = rel_codes[:, 0::4] / wx
        dy = rel_codes[:, 1::4] / wy
        dw = rel_codes[:, 2::4] / ww
        dh = rel_codes[:, 3::4] / wh

        # Prevent sending too large values into torch.exp()
        dw = flow.clamp(dw, max=self.bbox_xform_clip)
        dh = flow.clamp(dh, max=self.bbox_xform_clip)

        pred_ctr_x = dx * widths[:, None] + ctr_x[:, None]
        pred_ctr_y = dy * heights[:, None] + ctr_y[:, None]
        pred_w = flow.exp(dw) * widths[:, None]
        pred_h = flow.exp(dh) * heights[:, None]

        pred_boxes1 = pred_ctr_x - flow.tensor(0.5, dtype=pred_ctr_x.dtype, device=pred_w.device) * pred_w
        pred_boxes2 = pred_ctr_y - flow.tensor(0.5, dtype=pred_ctr_y.dtype, device=pred_h.device) * pred_h
        pred_boxes3 = pred_ctr_x + flow.tensor(0.5, dtype=pred_ctr_x.dtype, device=pred_w.device) * pred_w
        pred_boxes4 = pred_ctr_y + flow.tensor(0.5, dtype=pred_ctr_y.dtype, device=pred_h.device) * pred_h
        pred_boxes = flow.stack((pred_boxes1, pred_boxes2, pred_boxes3, pred_boxes4), dim=2).flatten(1)
        return pred_boxes


class Matcher(object):
    """
    This class assigns to each predicted "element" (e.g., a box) a ground-truth
    element. Each predicted element will have exactly zero or one matches; each
    ground-truth element may be assigned to zero or more predicted elements.

    Matching is based on the MxN match_quality_matrix, that characterizes how well
    each (ground-truth, predicted)-pair match. For example, if the elements are
    boxes, the matrix may contain box IoU overlap values.

    The matcher returns a tensor of size N containing the index of the ground-truth
    element m that matches to prediction n. If there is no match, a negative value
    is returned.
    """

    BELOW_LOW_THRESHOLD = -1
    BETWEEN_THRESHOLDS = -2

    __annotations__ = {
        'BELOW_LOW_THRESHOLD': int,
        'BETWEEN_THRESHOLDS': int,
    }

    def __init__(self, high_threshold, low_threshold, allow_low_quality_matches=False):
        # type: (float, float, bool) -> None
        """
        Args:
            high_threshold (float): quality values greater than or equal to
                this value are candidate matches.
            low_threshold (float): a lower quality threshold used to stratify
                matches into three levels:
                1) matches >= high_threshold
                2) BETWEEN_THRESHOLDS matches in [low_threshold, high_threshold)
                3) BELOW_LOW_THRESHOLD matches in [0, low_threshold)
            allow_low_quality_matches (bool): if True, produce additional matches
                for predictions that have only low-quality match candidates. See
                set_low_quality_matches_ for more details.
        """
        self.BELOW_LOW_THRESHOLD = -1
        self.BETWEEN_THRESHOLDS = -2
        assert low_threshold <= high_threshold
        self.high_threshold = high_threshold
        self.low_threshold = low_threshold
        self.allow_low_quality_matches = allow_low_quality_matches

    def __call__(self, match_quality_matrix):
        """
        Args:
            match_quality_matrix (Tensor[float]): an MxN tensor, containing the
            pairwise quality between M ground-truth elements and N predicted elements.

        Returns:
            matches (Tensor[int64]): an N tensor where N[i] is a matched gt in
            [0, M - 1] or a negative value indicating that prediction i could not
            be matched.
        """
        if match_quality_matrix.numel() == 0:
            # empty targets or proposals not supported during training
            if match_quality_matrix.shape[0] == 0:
                raise ValueError(
                    "No ground-truth boxes available for one of the images "
                    "during training")
            else:
                raise ValueError(
                    "No proposal boxes available for one of the images "
                    "during training")

        # match_quality_matrix is M (gt) x N (predicted)
        # Max over gt elements (dim 0) to find best gt candidate for each prediction
        matched_vals = match_quality_matrix.max(dim=0)
        matches = match_quality_matrix.argmax(dim=0).to(dtype=flow.int32)
        if self.allow_low_quality_matches:
            all_matches = matches.clone()
        else:
            all_matches = None

        # Assign candidate matches with low quality to negative (unassigned) values
        below_low_threshold = matched_vals < self.low_threshold
        # between_thresholds = (matched_vals >= self.low_threshold) & (
        #     matched_vals < self.high_threshold
        # )
        between_thresholds = (matched_vals >= self.low_threshold).mul(
                matched_vals < self.high_threshold
            )
        for idx in below_low_threshold.argwhere():
                matches[idx.numpy().item()] = flow.tensor(self.BELOW_LOW_THRESHOLD, dtype = matches.dtype, device = matches.device)
        for idx in between_thresholds.argwhere():
                matches[idx.numpy().item()] = flow.tensor(self.BETWEEN_THRESHOLDS, dtype = matches.dtype, device = matches.device)
        # matches[below_low_threshold] = self.BELOW_LOW_THRESHOLD
        # matches[between_thresholds] = self.BETWEEN_THRESHOLDS

        if self.allow_low_quality_matches:
            assert all_matches is not None
            self.set_low_quality_matches_(matches, all_matches, match_quality_matrix)

        return matches

    def set_low_quality_matches_(self, matches, all_matches, match_quality_matrix):
        """
        Produce additional matches for predictions that have only low-quality matches.
        Specifically, for each ground-truth find the set of predictions that have
        maximum overlap with it (including ties); for each prediction in that set, if
        it is unmatched, then match it to the ground-truth with which it has the highest
        quality value.
        """
        # For each gt, find the prediction with which it has highest quality
        highest_quality_foreach_gt = match_quality_matrix.max(dim=1)
        # Find highest quality match available, even if it is low, including ties
        # gt_pred_pairs_of_highest_quality = flow.where(
        #     match_quality_matrix == highest_quality_foreach_gt[:, None]
        # )
        gt_pred_pairs_of_highest_quality = flow.argwhere(flow.eq(match_quality_matrix, highest_quality_foreach_gt[:, None]))
        # Example gt_pred_pairs_of_highest_quality:
        #   tensor([[    0, 39796],
        #           [    1, 32055],
        #           [    1, 32070],
        #           [    2, 39190],
        #           [    2, 40255],
        #           [    3, 40390],
        #           [    3, 41455],
        #           [    4, 45470],
        #           [    5, 45325],
        #           [    5, 46390]])
        # Each row is a (gt index, prediction index)
        # Note how gt items 1, 2, 3, and 5 each have two ties

        pred_inds_to_update = gt_pred_pairs_of_highest_quality[1]
        for idx in pred_inds_to_update:
            matches[idx] = all_matches[idx]


def smooth_l1_loss(input, target, beta: float = 1. / 9, size_average: bool = True):
    """
    very similar to the smooth_l1_loss from pytorch, but with
    the extra beta parameter
    """
    n = flow.abs(input - target)
    cond = n < beta
    loss = flow.where(cond, 0.5 * n ** 2 / beta, n - 0.5 * beta)
    if size_average:
        return loss.mean()
    return loss.sum()


def overwrite_eps(model, eps):
    """
    This method overwrites the default eps values of all the
    FrozenBatchNorm2d layers of the model with the provided value.
    This is necessary to address the BC-breaking change introduced
    by the bug-fix at pytorch/vision#2933. The overwrite is applied
    only when the pretrained weights are loaded to maintain compatibility
    with previous versions.

    Args:
        model (nn.Module): The model on which we perform the overwrite.
        eps (float): The new value of eps.
    """
    for module in model.modules():
        if isinstance(module, FrozenBatchNorm2d):
            module.eps = eps
