# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import oneflow as flow
from oneflow import nn, Tensor

from ops import boxes as box_ops

from utils import detection_utils as det_utils
from utils.image_list import ImageList

from typing import List, Optional, Dict, Tuple

# Import AnchorGenerator to keep compatibility.
from utils.anchor_utils import AnchorGenerator


# @torch.jit.unused
# def _onnx_get_num_anchors_and_pre_nms_top_n(ob, orig_pre_nms_top_n):
#     # type: (Tensor, int) -> Tuple[int, int]
#     from torch.onnx import operators
#     num_anchors = operators.shape_as_tensor(ob)[1].unsqueeze(0)
#     pre_nms_top_n = torch.min(torch.cat(
#         (torch.tensor([orig_pre_nms_top_n], dtype=num_anchors.dtype),
#          num_anchors), 0))
#
#     return num_anchors, pre_nms_top_n


class RPNHead(nn.Module):
    """
    Adds a simple RPN Head with classification and regression heads

    Args:
        in_channels (int): number of channels of the input feature
        num_anchors (int): number of anchors to be predicted
    """

    def __init__(self, in_channels, num_anchors):
        super(RPNHead, self).__init__()
        self.conv = nn.Conv2d(
            in_channels, in_channels, kernel_size=3, stride=1, padding=1
        )
        self.cls_logits = nn.Conv2d(in_channels, num_anchors, kernel_size=1, stride=1)
        self.bbox_pred = nn.Conv2d(
            in_channels, num_anchors * 4, kernel_size=1, stride=1
        )

        for layer in self.children():
            nn.init.normal_(layer.weight, std=0.01)
            nn.init.constant_(layer.bias, 0)

    def forward(self, x):
        # type: (List[Tensor]) -> Tuple[List[Tensor], List[Tensor]]
        logits = []
        bbox_reg = []
        for feature in x:
            # t = nn.ReLU((self.conv(feature)))
            t = nn.ReLU()(self.conv(feature))
            logits.append(self.cls_logits(t))
            bbox_reg.append(self.bbox_pred(t))
        return logits, bbox_reg


def permute_and_flatten(layer, N, A, C, H, W):
    # type: (Tensor, int, int, int, int, int) -> Tensor
    layer = layer.view(N, -1, C, H, W)
    layer = layer.permute(0, 3, 4, 1, 2)
    layer = layer.reshape(N, -1, C)
    return layer


def concat_box_prediction_layers(box_cls, box_regression):
    # type: (List[Tensor], List[Tensor]) -> Tuple[Tensor, Tensor]
    box_cls_flattened = []
    box_regression_flattened = []
    # for each feature level, permute the outputs to make them be in the
    # same format as the labels. Note that the labels are computed for
    # all feature levels concatenated, so we keep the same representation
    # for the objectness and the box_regression
    for box_cls_per_level, box_regression_per_level in zip(
            box_cls, box_regression
    ):
        N, AxC, H, W = box_cls_per_level.shape
        Ax4 = box_regression_per_level.shape[1]
        A = Ax4 // 4
        C = AxC // A
        box_cls_per_level = permute_and_flatten(
            box_cls_per_level, N, A, C, H, W
        )
        box_cls_flattened.append(box_cls_per_level)

        box_regression_per_level = permute_and_flatten(
            box_regression_per_level, N, A, 4, H, W
        )
        box_regression_flattened.append(box_regression_per_level)
    # concatenate on the first dimension (representing the feature levels), to
    # take into account the way the labels were generated (with all feature maps
    # being concatenated as well)
    box_cls = flow.cat(box_cls_flattened, dim=1).flatten(0, -2)
    box_regression = flow.cat(box_regression_flattened, dim=1).reshape(-1, 4)
    # oneflow._oneflow_internal.Tensor -> flow.Tensor
    # box_regression = flow.Tensor(*box_regression)
    return box_cls, box_regression


class RegionProposalNetwork(nn.Module):
    """
    Implements Region Proposal Network (RPN).

    Args:
        anchor_generator (AnchorGenerator): module that generates the anchors for a set of feature
            maps.
        head (nn.Module): module that computes the objectness and regression deltas
        fg_iou_thresh (float): minimum IoU between the anchor and the GT box so that they can be
            considered as positive during training of the RPN.
        bg_iou_thresh (float): maximum IoU between the anchor and the GT box so that they can be
            considered as negative during training of the RPN.
        batch_size_per_image (int): number of anchors that are sampled during training of the RPN
            for computing the loss
        positive_fraction (float): proportion of positive anchors in a mini-batch during training
            of the RPN
        pre_nms_top_n (Dict[int]): number of proposals to keep before applying NMS. It should
            contain two fields: training and testing, to allow for different values depending
            on training or evaluation
        post_nms_top_n (Dict[int]): number of proposals to keep after applying NMS. It should
            contain two fields: training and testing, to allow for different values depending
            on training or evaluation
        nms_thresh (float): NMS threshold used for postprocessing the RPN proposals

    """
    __annotations__ = {
        'box_coder': det_utils.BoxCoder,
        'proposal_matcher': det_utils.Matcher,
        'fg_bg_sampler': det_utils.BalancedPositiveNegativeSampler,
        'pre_nms_top_n': Dict[str, int],
        'post_nms_top_n': Dict[str, int],
    }

    def __init__(self,
                 anchor_generator,
                 head,
                 #
                 fg_iou_thresh, bg_iou_thresh,
                 batch_size_per_image, positive_fraction,
                 #
                 pre_nms_top_n, post_nms_top_n, nms_thresh, score_thresh=0.0):
        super(RegionProposalNetwork, self).__init__()
        self.anchor_generator = anchor_generator
        self.head = head
        self.box_coder = det_utils.BoxCoder(weights=(1.0, 1.0, 1.0, 1.0))

        # used during training
        self.box_similarity = box_ops.box_iou

        # self.proposal_matcher = det_utils.Matcher(
        #     fg_iou_thresh,
        #     bg_iou_thresh,
        #     allow_low_quality_matches=True,
        # )
        self.proposal_matcher = det_utils.Matcher(
            fg_iou_thresh,
            bg_iou_thresh,
            allow_low_quality_matches=False,
        )

        self.fg_bg_sampler = det_utils.BalancedPositiveNegativeSampler(
            batch_size_per_image, positive_fraction
        )
        # used during testing
        self._pre_nms_top_n = pre_nms_top_n
        self._post_nms_top_n = post_nms_top_n
        self.nms_thresh = nms_thresh
        self.score_thresh = score_thresh
        self.min_size = 1e-3

    def pre_nms_top_n(self):
        if self.training:
            return self._pre_nms_top_n['training']
        return self._pre_nms_top_n['testing']

    def post_nms_top_n(self):
        if self.training:
            return self._post_nms_top_n['training']
        return self._post_nms_top_n['testing']

    def assign_targets_to_anchors(self, anchors, targets):
        # type: (List[Tensor], List[Dict[str, Tensor]]) -> Tuple[List[Tensor], List[Tensor]]
        labels = []
        matched_gt_boxes = []
        for anchors_per_image, targets_per_image in zip(anchors, targets):
            gt_boxes = targets_per_image["boxes"]
            if gt_boxes.numel() == 0:
                # Background image (negative example)
                device = anchors_per_image.device
                matched_gt_boxes_per_image = flow.zeros(anchors_per_image.shape, dtype=flow.float32, device=device)
                labels_per_image = flow.zeros((anchors_per_image.shape[0],), dtype=flow.int32, device=device)
            else:
                anchors_per_image = anchors_per_image.view(-1, 4)
                match_quality_matrix = self.box_similarity(gt_boxes, anchors_per_image)
                matched_idxs = self.proposal_matcher(match_quality_matrix)
                # get the targets corresponding GT for each proposal
                # NB: need to clamp the indices because we can have a single
                # GT in the image, and matched_idxs can be -2, which goes
                # out of bounds
                # matched_gt_boxes_per_image = gt_boxes[matched_idxs.clamp(min=0)]
                matched_gt_boxes_per_image = flow.zeros((matched_idxs.shape[0], 4), device=gt_boxes.device,
                                                        dtype=gt_boxes.dtype)
                for idx, val in enumerate(matched_idxs.clamp(min=0)):
                    matched_gt_boxes_per_image[idx, :] = gt_boxes[val.numpy().item()]

                labels_per_image = (matched_idxs >= 0).to(dtype=flow.int32)
                for idx in labels_per_image.argwhere():
                    labels_per_image[idx.numpy().item()] = matched_idxs[idx.numpy().item()]
                # labels_per_image = labels_per_image.to(dtype=flow.float32)

                # Background (negative examples)
                bg_indices = flow.eq(matched_idxs, self.proposal_matcher.BELOW_LOW_THRESHOLD).argwhere()
                for idx in bg_indices:
                    labels_per_image[idx.numpy().item()] = flow.tensor(0, device = labels_per_image.device, dtype=labels_per_image.dtype)

                # discard indices that are between thresholds
                inds_to_discard = flow.eq(matched_idxs, self.proposal_matcher.BETWEEN_THRESHOLDS).argwhere()
                for idx in inds_to_discard:
                    labels_per_image[idx.numpy().item()] = flow.tensor(-1, device = labels_per_image.device, dtype=labels_per_image.dtype)

            labels.append(labels_per_image)
            matched_gt_boxes.append(matched_gt_boxes_per_image)
        return labels, matched_gt_boxes

    def _get_top_n_idx(self, objectness, num_anchors_per_level):
        # type: (Tensor, List[int]) -> Tensor
        # r = []
        # offset = 0
        #
        # # if torchvision._is_tracing():
        # #     num_anchors, pre_nms_top_n = _onnx_get_num_anchors_and_pre_nms_top_n(ob, self.pre_nms_top_n())
        # # else:
        #
        # # for ob in objectness.split(num_anchors_per_level, 1):
        # for ob in objectness:
        #     ob = ob.unsqueeze(dim=0)
        #     num_anchors = ob.shape[1]
        #     pre_nms_top_n = min(self.pre_nms_top_n(), num_anchors)
        #     _, top_n_idx = ob.topk(pre_nms_top_n, dim=1)
        #     r.append(top_n_idx + offset)
        #     offset += num_anchors
        # return flow.cat(r, dim=1)

        r = []
        for ob in objectness:
            ob = ob.unsqueeze(dim=0)
            num_anchors = ob.shape[1]
            pre_nms_top_n = min(self.pre_nms_top_n(), num_anchors)
            _, top_n_idx = ob.topk(pre_nms_top_n, dim=1)
            r.append(top_n_idx)

        return r

    def filter_proposals(self, proposals, objectness, image_shapes, num_anchors_per_level):
        # type: (Tensor, Tensor, List[Tuple[int, int]], List[int]) -> Tuple[List[Tensor], List[Tensor]]
        num_images = proposals.shape[0]
        device = proposals.device
        # do not backprop throught objectness
        objectness = objectness.detach()
        objectness = objectness.reshape(num_images, -1)

        levels = [
            # flow.full((n,), idx, dtype=torch.int64, device=device)
            flow.ones((n,), dtype=flow.float32, device=device) * idx
            for idx, n in enumerate(num_anchors_per_level)
        ]
        levels = flow.cat(levels, 0)
        # print(levels.shape, levels.dtype, levels.device, objectness.shape, objectness.dtype, objectness.device)
        levels = (levels.reshape(1, -1)).expand(*objectness.shape)

        # select top_n boxes independently per level before applying nms
        top_n_idx = self._get_top_n_idx(objectness, num_anchors_per_level)
        print("top_k done", num_images, device)

        # TODO:
        # advance indexing
        # image_range = flow.arange(num_images, device=device, dtype=flow.int64)
        # print("image_range", image_range)
        # # batch_idx = image_range[:, None]
        # batch_idx = image_range.unsqueeze(dim=1)
        # print("batch_idx", batch_idx, "top_n_idx", top_n_idx.shape, "objectness", objectness.shape)
        # objectness = objectness[batch_idx, top_n_idx]
        # print(objectness)
        # levels = levels[batch_idx, top_n_idx]
        # proposals = proposals[batch_idx, top_n_idx]
        # print(top_n_idx[0].shape)

        tmp_levels = flow.zeros_like(levels)
        tmp_proposals = flow.zeros_like(proposals)
        for i in range(num_images):
            for j in range(top_n_idx[i].shape[1]):
                # if j >= self.pre_nms_top_n():
                #     offset = j // self.pre_nms_top_n()
                #     offset = offset * objectness.shape[1]
                # else:
                #     offset = 0
                # print(i, j, top_n_idx_numpy[0, j], type(top_n_idx_numpy[0, j].item()))
                top_n_idx_numpy = top_n_idx[i].numpy()
                tmp_levels[i, j] = levels[i, top_n_idx_numpy[0, j].item()]
                tmp_proposals[i, j] = proposals[i, top_n_idx_numpy[0, j].item()]
        levels = tmp_levels[:, :top_n_idx[0].shape[1]]
        proposals = tmp_proposals[:, :top_n_idx[0].shape[1]]

        # print(levels)
        # tmp = flow.zeros_like(proposals)
        # for i in range(num_images):
        #     for j in range(top_n_idx[0].shape[0]):
        #

        # print(proposals)
        objectness_prob = flow.sigmoid(objectness)

        final_boxes = []
        final_scores = []
        for boxes, scores, lvl, img_shape in zip(proposals, objectness_prob, levels, image_shapes):
            boxes = box_ops.clip_boxes_to_image(boxes, img_shape)

            # remove small boxes
            keep = box_ops.remove_small_boxes(boxes, self.min_size)
            # TODO:advance index
            # boxes, scores, lvl = boxes[keep], scores[keep], lvl[keep]
            tmp_boxes = flow.zeros_like(boxes)
            tmp_scores = flow.zeros_like(scores)
            tmp_lvl = flow.zeros_like(lvl)
            keep_numpy = keep.numpy()
            for i in range(keep.shape[0]):
                tmp_boxes[i], tmp_scores[i], tmp_lvl[i] = boxes[keep_numpy[i].item()], scores[keep_numpy[i].item()], \
                                                          lvl[keep_numpy[i].item()]
            boxes = tmp_boxes[:keep.shape[0]]
            scores = tmp_scores[:keep.shape[0]]
            lvl = tmp_lvl[:keep.shape[0]]

            # remove low scoring boxes
            # use >= for Backwards compatibility
            keep = flow.argwhere(scores >= self.score_thresh)
            keep_numpy = keep.numpy()
            # TODO:advance index
            # boxes, scores, lvl = boxes[keep], scores[keep], lvl[keep]
            for i in range(keep.shape[0]):
                tmp_boxes[i], tmp_scores[i], tmp_lvl[i] = boxes[keep_numpy[i].item()], scores[keep_numpy[i].item()], \
                                                          lvl[keep_numpy[i].item()]
            boxes = tmp_boxes[:keep.shape[0]]
            scores = tmp_scores[:keep.shape[0]]
            lvl = tmp_lvl[:keep.shape[0]]

            # non-maximum suppression, independently done per level
            boxes = boxes.to('cuda')
            scores = scores.to('cuda')
            keep = box_ops.batched_nms(boxes, scores, lvl, self.nms_thresh)

            # keep only topk scoring predictions
            keep = keep[:self.post_nms_top_n()]
            # TODO:advance index
            tmp_boxes = flow.zeros_like(boxes)
            tmp_scores = flow.zeros_like(scores)
            keep_numpy = keep.numpy()
            for i in range(keep.shape[0]):
                tmp_boxes[i], tmp_scores[i], = boxes[keep_numpy[i].item()], scores[keep_numpy[i].item()]
            boxes, scores = tmp_boxes, tmp_scores
            final_boxes.append(boxes)
            final_scores.append(scores)
        return final_boxes, final_scores

    def compute_loss(self, objectness, pred_bbox_deltas, labels, regression_targets):
        # type: (Tensor, Tensor, List[Tensor], List[Tensor]) -> Tuple[Tensor, Tensor]
        """
        Args:
            objectness (Tensor)
            pred_bbox_deltas (Tensor)
            labels (List[Tensor])
            regression_targets (List[Tensor])

        Returns:
            objectness_loss (Tensor)
            box_loss (Tensor)
        """

        sampled_pos_inds, sampled_neg_inds = self.fg_bg_sampler(labels)
        sampled_pos_inds = flow.argwhere(flow.cat(sampled_pos_inds, dim=0))[0]
        sampled_neg_inds = flow.argwhere(flow.cat(sampled_neg_inds, dim=0))[0]

        sampled_inds = flow.cat([sampled_pos_inds, sampled_neg_inds], dim=0)

        objectness = objectness.flatten()

        labels = flow.cat(labels, dim=0)
        regression_targets = flow.cat(regression_targets, dim=0)

        box_loss = det_utils.smooth_l1_loss(
            pred_bbox_deltas[sampled_pos_inds.numpy().item()],
            regression_targets[sampled_pos_inds.numpy().item()],
            beta=1 / 9,
            size_average=False,
        ) / (sampled_inds.numel())


        objectness_loss = nn.BCEWithLogitsLoss(
            objectness[sampled_inds], labels[sampled_inds]
        )

        return objectness_loss, box_loss

    def forward(self,
                images,  # type: ImageList
                features,  # type: Dict[str, Tensor]
                targets=None  # type: Optional[List[Dict[str, Tensor]]]
                ):
        # type: (...) -> Tuple[List[Tensor], Dict[str, Tensor]]
        """
        Args:
            images (ImageList): images for which we want to compute the predictions
            features (OrderedDict[Tensor]): features computed from the images that are
                used for computing the predictions. Each tensor in the list
                correspond to different feature levels
            targets (List[Dict[Tensor]]): ground-truth boxes present in the image (optional).
                If provided, each element in the dict should contain a field `boxes`,
                with the locations of the ground-truth boxes.

        Returns:
            boxes (List[Tensor]): the predicted boxes from the RPN, one Tensor per
                image.
            losses (Dict[Tensor]): the losses for the model during training. During
                testing, it is an empty dict.
        """
        # RPN uses all feature maps that are available
        features = list(features.values())
        objectness, pred_bbox_deltas = self.head(features)
        anchors = self.anchor_generator(images, features)
        num_images = len(anchors)
        num_anchors_per_level_shape_tensors = [o[0].shape for o in objectness]
        num_anchors_per_level = [s[0] * s[1] * s[2] for s in num_anchors_per_level_shape_tensors]
        objectness, pred_bbox_deltas = \
            concat_box_prediction_layers(objectness, pred_bbox_deltas)
        # apply pred_bbox_deltas to anchors to obtain the decoded proposals
        # note that we detach the deltas because Faster R-CNN do not backprop through
        # the proposals
        proposals = self.box_coder.decode(pred_bbox_deltas.detach(), anchors)
        proposals = proposals.view(num_images, -1, 4)
        boxes, scores = self.filter_proposals(proposals, objectness, images.image_sizes, num_anchors_per_level)

        losses = {}
        if self.training:
            assert targets is not None
            labels, matched_gt_boxes = self.assign_targets_to_anchors(anchors, targets)
            regression_targets = self.box_coder.encode(matched_gt_boxes, anchors)
            loss_objectness, loss_rpn_box_reg = self.compute_loss(
                objectness, pred_bbox_deltas, labels, regression_targets)
            losses = {
                "loss_objectness": loss_objectness,
                "loss_rpn_box_reg": loss_rpn_box_reg,
            }
        return boxes, losses
