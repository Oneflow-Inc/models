import oneflow as flow

import argparse
import numpy as np
import time

from models.inceptionv3 import inception_v3
from utils.imagenet1000_clsidx_to_labels import clsidx_2_labels
from utils.numpy_data_utils import load_image


def _parse_args():
    parser = argparse.ArgumentParser("flags for test inceptionv3")
    parser.add_argument(
        "--model_path",
        type=str,
        default="./inceptionv3_oneflow_model",
        help="model path",
    )
    parser.add_argument("--image_path", type=str, default="", help="input image path")
    return parser.parse_args()


def main(args):

    start_t = time.time()
    inceptionv3_module = inception_v3()
    end_t = time.time()
    print("init time : {}".format(end_t - start_t))

    start_t = time.time()
    pretrain_models = flow.load(args.model_path)
    inceptionv3_module.load_state_dict(pretrain_models)
    end_t = time.time()
    print("load params time : {}".format(end_t - start_t))

    inceptionv3_module.eval()
    inceptionv3_module.to("cuda")

    start_t = time.time()
    image = load_image(args.image_path)
    image = flow.Tensor(image, device=flow.device("cuda"))
    predictions, aux_predictions = inceptionv3_module(image)
    predictions = predictions.softmax()
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
