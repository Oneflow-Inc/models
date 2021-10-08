import oneflow as flow

import argparse
import numpy as np
import time

from model.build_model import build_model
from utils.imagenet1000_clsidx_to_labels import clsidx_2_labels
from utils.numpy_data_utils import load_image

def _parse_args():
    parser = argparse.ArgumentParser("flags for test ViT")
    parser.add_argument(
        "--model_path",
        type=str,
        default="./vit_b_16_384",
        help="model path",
    )
    parser.add_argument(
        "--model_arch",
        type=str,
        default="vit_b_16_384",
        choices=[
            "vit_b_16_224",
            "vit_b_16_384",
            "vit_b_32_224",
            "vit_b_32_384",
            "vit_l_16_224",
            "vit_l_16_384",
        ],
        help="model architecture",
    )
    parser.add_argument("--image_size", type=int, default=384, help="input image size")
    parser.add_argument("--image_path", type=str, default="", help="input image path")
    return parser.parse_args()


def main(args):

    start_t = time.time()
    model = build_model(args)
    end_t = time.time()
    print("init time : {}".format(end_t - start_t))

    start_t = time.time()
    pretrain_models = flow.load(args.model_path)
    model.load_state_dict(pretrain_models)
    end_t = time.time()
    print("load params time : {}".format(end_t - start_t))

    model.eval()
    model.to("cuda")

    start_t = time.time()
    image = load_image(args.image_path, image_size=(args.image_size, args.image_size))
    image = flow.Tensor(image, device=flow.device("cuda"))
    predictions = model(image).softmax()
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
