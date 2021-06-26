import oneflow.experimental as flow

import argparse
import numpy as np
import time

from models.repvgg import create_RepVGG_A0
from utils.imagenet1000_clsidx_to_labels import clsidx_2_labels
from utils.numpy_data_utils import load_image

def _parse_args():
    parser = argparse.ArgumentParser("flags for test repVGGA0")
    parser.add_argument(
        "--model_path", type=str, default="./RepVGG-A0-train", help="model path"
    )
    parser.add_argument(
        "--image_path", type=str, default="", help="input image path"
    )
    return parser.parse_args()

def main(args):
    flow.env.init()
    flow.enable_eager_execution()

    start_t = time.time()
    repVGGA0 = create_RepVGG_A0()
    end_t = time.time()
    print('init time : {}'.format(end_t - start_t))


    start_t = time.time()
    pretrain_models = flow.load(args.model_path)
    repVGGA0.load_state_dict(pretrain_models)
    end_t = time.time()
    print('load params time : {}'.format(end_t - start_t))

    repVGGA0.eval()
    repVGGA0.to("cuda")

    start_t = time.time()
    image = load_image(args.image_path)
    image = flow.Tensor(image, device=flow.device("cuda"))
    predictions = repVGGA0(image).softmax()
    predictions = predictions.numpy()
    end_t = time.time()
    print('infer time : {}'.format(end_t - start_t))
    clsidx = np.argmax(predictions)
    print("predict prob: %f, class name: %s" % (np.max(predictions), clsidx_2_labels[clsidx]))

if __name__ == "__main__":
    args = _parse_args()
    main(args)
