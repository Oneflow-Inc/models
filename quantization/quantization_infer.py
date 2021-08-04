import oneflow as flow

import argparse
import numpy as np
import time

from models.q_alexnet import QuantizationAlexNet
from utils.imagenet1000_clsidx_to_labels import clsidx_2_labels
from utils.numpy_data_utils import load_image


def _parse_args():
    parser = argparse.ArgumentParser("flags for test alexnet")
    parser.add_argument(
        "--model_path", type=str, default="./alexnet_oneflow_model", help="model path"
    )
    parser.add_argument("--image_path", type=str, default="", help="input image path")
    parser.add_argument("--quantization_bit", type=int, default=8, help="quantization bit")
    parser.add_argument("--quantization_scheme", type=str, default="symmetric", help="quantization scheme")
    parser.add_argument("--quantization_formula", type=str, default="google", help="quantization formula")
    parser.add_argument("--per_layer_quantization", type=bool, default=True, help="per_layer_quantization")
    return parser.parse_args()


def main(args):
    start_t = time.time()
    alexnet_module = QuantizationAlexNet()
    alexnet_module.quantize(quantization_bit=args.quantization_bit, quantization_scheme=args.quantization_scheme, 
                                quantization_formula=args.quantization_formula, per_layer_quantization=args.per_layer_quantization)
    end_t = time.time()
    print("init time : {}".format(end_t - start_t))

    start_t = time.time()
    pretrain_models = flow.load(args.model_path)
    alexnet_module.load_state_dict(pretrain_models)
    end_t = time.time()
    print("load params time : {}".format(end_t - start_t))

    alexnet_module.to("cuda")

    start_t = time.time()
    image = load_image(args.image_path)
    image = flow.Tensor(image, device=flow.device("cuda"))
    predictions = alexnet_module.quantize_forward(image).softmax()
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
