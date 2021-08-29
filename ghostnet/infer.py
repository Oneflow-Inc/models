import oneflow as flow

import argparse
import numpy as np
import time

from models.ghostnet import ghostnet
from utils.imagenet1000_clsidx_to_labels import clsidx_2_labels
from utils.numpy_data_utils import load_image

model_dict = {"ghostnet": ghostnet}


def _parse_args():
    parser = argparse.ArgumentParser("flags for test ghostnet")
    parser.add_argument(
        "--model_path",
        type=str,
        default="ghostnet_imagenet_pretrain_model/",
        help="model path",
    )
    parser.add_argument("--image_path", type=str, default="", help="input image path")
    parser.add_argument(
        "--model", type=str, default="ghostnet", help="choose from ghostnet",
    )
    return parser.parse_args()


def main(args):
    assert args.model in model_dict
    print("Predicting using", args.model, "...")

    start_t = time.time()
    net_module = model_dict[args.model]()
    end_t = time.time()
    print("init time : {}".format(end_t - start_t))

    start_t = time.time()
    pretrain_models = flow.load(args.model_path)
    net_module.load_state_dict(pretrain_models)
    end_t = time.time()
    print("load params time : {}".format(end_t - start_t))

    net_module.eval()
    net_module.to("cuda")

    start_t = time.time()
    image = load_image(args.image_path)
    image = flow.Tensor(image, device=flow.device("cuda"))
    predictions = net_module(image).softmax()
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
