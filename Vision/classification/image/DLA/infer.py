import oneflow as flow
import argparse
import numpy as np
import time
from models.dla import DLA
from utils.clsidx_to_labels import clsidx_2_labels
from utils.numpy_data_utils import load_image


def _parse_args():
    parser = argparse.ArgumentParser("flags for test dla")
    parser.add_argument(
        "--model_path",
        type=str,
        default="./pretrain_model/epoch_1396_val_acc_0.921875",
        help="model path",
    )
    parser.add_argument(
        "--image_path", type=str, default="./data/bus.jpg", help="input image path"
    )
    return parser.parse_args()


def main(args):
    start_t = time.time()
    dla_module = DLA(
        num_classes=10, levels=[1, 1, 1, 2, 2, 1], channels=[16, 32, 64, 128, 256, 512]
    )
    end_t = time.time()
    print("init time : {}".format(end_t - start_t))
    start_t = time.time()
    pretrain_models = flow.load(args.model_path)
    dla_module.load_state_dict(pretrain_models)
    end_t = time.time()
    print("load params time : {}".format(end_t - start_t))
    dla_module.eval()
    dla_module.to("cuda")
    start_t = time.time()
    image = load_image(args.image_path)
    image = flow.Tensor(image, device=flow.device("cuda"))
    logits = dla_module(image)
    predictions = logits.softmax()
    predictions = predictions.numpy()
    end_t = time.time()
    print("infer time : {}".format(end_t - start_t))
    clsidx = np.argmax(predictions)
    print(
        "predict prob: %f, class name: %s"
        % (np.max(predictions), clsidx_2_labels[clsidx])
    )


if __name__ == "__main__":
    args = _parse_args()
    main(args)
