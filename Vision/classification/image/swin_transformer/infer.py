import oneflow as flow

import argparse
import numpy as np
import time

from models.swin_transformer import create_swin_transformer
from utils.imagenet1000_clsidx_to_labels import clsidx_2_labels
from utils.numpy_data_utils import load_image


def _parse_args():
    parser = argparse.ArgumentParser("flags for test Swin Transformer")
    parser.add_argument(
        "--model_path", type=str, default="./swin_tiny_patch4_window7_224_oneflow_model", help="model path"
    )
    parser.add_argument("--image_path", type=str, default="", help="input image path")
    return parser.parse_args()


def main(args):

    start_t = time.time()
    swin_transformer = create_swin_transformer()
    end_t = time.time()
    print("init time : {}".format(end_t - start_t))

    start_t = time.time()
    pretrain_models = flow.load(args.model_path)
    swin_transformer.load_state_dict(pretrain_models)
    end_t = time.time()
    print("load params time : {}".format(end_t - start_t))

    swin_transformer.eval()
    swin_transformer.to("cuda")

    start_t = time.time()
    image = load_image(args.image_path)
    image = flow.Tensor(image, device=flow.device("cuda"))
    predictions = swin_transformer(image).softmax()
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
