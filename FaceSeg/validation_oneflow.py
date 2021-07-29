import oneflow as flow
import oneflow.nn as nn
import numpy as np
from models.LinkNet34 import LinkNet34
import argparse
from utils.numpy_data_utils import face_seg

# arguments


def parse_args():
    parser = argparse.ArgumentParser(description='Face segmentation')
    # for oneflow
    parser.add_argument("--gpu_num_per_node", type=int,
                        default=1, required=False)
    parser.add_argument("--model_load_dir", type=str, default='./linknet_oneflow_model',
                        required=False, help="model load directory")
    parser.add_argument("--train_dataset_path", type=str,
                        default='./faceseg_data/', required=False, help="dataset root directory")
    parser.add_argument("--val_dataset_path", type=str, default='./faceseg_data/',
                        required=False, help="dataset root directory")
    parser.add_argument("--train_batch_size", type=int,
                        default=16, required=False)
    parser.add_argument("--val_batch_size", type=int,
                        default=16, required=False)
    parser.add_argument("--jaccard_weight", type=float, default=1, required=False,
                        help='jaccard weight for loss, a float between 0 and 1.')

    args = parser.parse_args()
    return args


# test config
args = parse_args()
batch_size = args.train_batch_size
model_pth = args.model_load_dir


class LossBinary:
    def __init__(self, jaccard_weight=0):
        self.nll_loss = nn.BCEWithLogitsLoss(reduction='mean')
        self.jaccard_weight = jaccard_weight

    def __call__(self, outputs, targets):
        loss = (1 - self.jaccard_weight) * self.nll_loss(outputs, targets)
        if self.jaccard_weight:
            eps = 1e-15
            jaccard_target = flow.Tensor(
                (targets.numpy() == 1)).to(flow.device('cuda'))
            jaccard_output = flow.sigmoid(outputs)
            intersection = (jaccard_output * jaccard_target).sum()
            union = jaccard_output.sum() + jaccard_target.sum()
            loss -= self.jaccard_weight * \
                flow.log((intersection + eps) / (union - intersection + eps))
        return loss


# evaluation the mIoU
class Criterion():

    def __init__(self, jaccard_weight=0):
        self.jaccard_weight = jaccard_weight
        self.num_classes = 2
        self.hist = np.zeros((self.num_classes, self.num_classes))

    '''
    Implementation by: https://github.com/LeeJunHyun/Image_Segmentation
    '''

    def get_miou(self, pred, target):
        # pred: output of network, shape of (batch_size, 1, img_size, img_size)
        # target: true mask, shape of (batch_size, 1, img_size, img_size)

        pred = np.reshape(pred, (batch_size, -1))
        pred = pred > 0.  # get the predict label, positive as label
        target = np.reshape(target, (batch_size, -1))
        inter = np.logical_and(pred, target,)
        union = np.logical_or(pred, target)
        # iou equation, add 1e-6 to avoid zero division
        iou_np = np.sum(inter, axis=-1) / (np.sum(union, axis=-1) + 1e-6)
        iou_np = np.mean(iou_np)
        return iou_np


def evaluate():
    # evaluate iou and loss of the model

    # load train and validate data
    train_data_loader = face_seg(
        args.train_dataset_path, batch_size, augmentation=None)
    val_data_loader = face_seg(
        args.val_dataset_path, batch_size, augmentation=None, training=False)

    # load model
    model = LinkNet34(pretrained=False)
    model.load_state_dict(flow.load(args.model_load_dir))
    model.to(flow.device('cuda'))

    # Eval on train data
    train_loss = 0
    miou = 0

    criterion = Criterion()
    Loss = LossBinary(jaccard_weight=1)
    for b in range(len(train_data_loader)):

        image_nd, label_nd = train_data_loader[b]
        image = flow.Tensor(image_nd)
        label = flow.Tensor(label_nd, dtype=flow.float32, requires_grad=False)
        image = image.to(flow.device('cuda'))
        label = label.to(flow.device('cuda'))
        logits = model(image)
        mask_probs_flat = flow.reshape(logits, shape=[-1])
        true_masks_flat = flow.reshape(label, shape=[-1])
        loss = Loss(mask_probs_flat, true_masks_flat)
        l = loss.numpy()[0]
        train_loss += l
        iou_np = criterion.get_miou(logits.numpy(), label.numpy())
        miou += iou_np

    miou = miou / (b+1)

    train_loss = train_loss / (b + 1)
    print("Train loss of model %s : %.3f" % (model_pth, train_loss))
    print("Train MIoU of model %s : %.3f " % (model_pth, miou * 100))

    # Evaluate on validation data
    val_loss = 0
    miou = 0

    for b in range(len(val_data_loader)):

        image_nd, label_nd = val_data_loader[b]
        image = flow.Tensor(image_nd)
        label = flow.Tensor(label_nd, dtype=flow.float32, requires_grad=False)
        image = image.to(flow.device('cuda'))
        label = label.to(flow.device('cuda'))
        logits = model(image)
        mask_probs_flat = flow.reshape(logits, shape=[-1])
        true_masks_flat = flow.reshape(label, shape=[-1])
        loss = Loss(mask_probs_flat, true_masks_flat)
        l = loss.numpy()[0]
        val_loss += l
        iou_np = criterion.get_miou(logits.numpy(), label.numpy())
        miou += iou_np

    miou = miou / (b+1)

    val_loss = val_loss / (b + 1)
    print("Val loss of model %s : %.3f" % (model_pth, val_loss))
    print("Val MIoU of model %s : %.3f " % (model_pth, miou * 100))


if __name__ == '__main__':
    evaluate()
