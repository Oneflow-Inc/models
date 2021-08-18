import os
import oneflow as flow
import argparse
import numpy as np
import time
from utils.data_utils import load_image
from utils.utils import to_numpy, to_tensor, save_images
from models.networks import Generator


def main(args):
    test_x, test_y = load_image(args.image_path)

    test_inp = to_tensor(test_x.astype(np.float32))
    test_target = to_tensor(test_y.astype(np.float32))

    generator = Generator().to("cuda")

    start_t = time.time()
    pretrain_model = flow.load(args.model_path)
    generator.load_state_dict(pretrain_model)
    end_t = time.time()
    print("load params time : {}".format(end_t - start_t))

    start_t = time.time()
    generator.eval()
    with flow.no_grad():
        gout = to_numpy(generator(test_inp), False)
    end_t = time.time()
    print("infer time : {}".format(end_t - start_t))

    # save images
    save_images(
        gout,
        test_inp.numpy(),
        test_target.numpy(),
        path=os.path.join("./testimage.png"),
        plot_size=1,
    )


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="oneflow PIX2PIX")
    parser.add_argument("--model_path", type=str, required=True, help="model path")
    parser.add_argument(
        "--image_path", type=str, required=True, help="input image path"
    )
    args = parser.parse_args()
    main(args)
