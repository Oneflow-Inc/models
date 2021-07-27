import oneflow as flow

import argparse
import numpy as np
import time

from models.shufflenetv2 import shufflenetv2_x0dot5, shufflenetv2_x1
from utils.imagenet1000_clsidx_to_labels import clsidx_2_labels
from utils.numpy_data_utils import load_image

model_dict = {
    "shufflenetv2_x0.5": shufflenetv2_x0dot5,
    "shufflenetv2_x1.0": shufflenetv2_x1,
}


def _parse_args():
    parser = argparse.ArgumentParser("flags for test shufflenet")
    parser.add_argument(
        "--model_path",
        type=str,
        default="shufflenetv2_imagenet_pretrain_model/",
        help="model path",
    )
    parser.add_argument("--image_path", type=str, default="", help="input image path")
    parser.add_argument(
        "--model",
        type=str,
        default="shufflenetv2_x0.5",
        help="choose from shufflenetv2_x0.5, shufflenetv2_x1.0",
    )
    return parser.parse_args()


def main(args):
    assert args.model in model_dict
    print("Predicting using", args.model, "...")
    
    
    start_t = time.time()
    shufflenet_module = model_dict[args.model]()
    end_t = time.time()
    print("init time : {}".format(end_t - start_t))

    start_t = time.time()
    pretrain_models = flow.load(args.model_path)
    shufflenet_module.load_state_dict(pretrain_models)
    end_t = time.time()
    print("load params time : {}".format(end_t - start_t))

    shufflenet_module.eval()
    shufflenet_module.to("cuda")

    start_t = time.time()
    image = load_image(args.image_path)
    image = flow.Tensor(image, device=flow.device("cuda"))
    predictions = shufflenet_module(image).softmax()
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
