import time
import argparse
from typing import Generator
import numpy as np
import oneflow.experimental as flow
from train_of_dcgan import Generator, to_tensor, to_numpy, save_images

def _parse_args():
    parser = argparse.ArgumentParser("flags for test DCGAN")
    parser.add_argument(
        "--model_path", type=str, default="./0.15dcgan/checkpoint/g_99", help="path to trained generator"
    )
    parser.add_argument(
        "--save_path", type=str, default="test_images.png", help="path to save"
    )
    parser.add_argument(
        "--batch_size", type=int, default=16, help="val batch size"
    )
    parser.add_argument('--no_cuda', action='store_true', default=False,
                        help='disables CUDA training')
    return parser.parse_args()

def main(args):
    flow.enable_eager_execution()

    device = 'cpu' if args.no_cuda else 'cuda'
    start_t = time.time()
    generator = Generator().to(device)
    end_t = time.time()
    print('init time : {}'.format(end_t - start_t))

    start_t = time.time()
    pretrain_model = flow.load(args.model_path)
    generator.load_state_dict(pretrain_model)
    end_t = time.time()
    print('load params time : {}'.format(end_t - start_t))

    generator.eval()

    start_t = time.time()
    z = to_tensor(
            np.random.normal(0, 1, size=(args.batch_size, 100)), False).to(device)
    predictions = to_numpy(generator(z), False)
    end_t = time.time()
    print('infer time : {}'.format(end_t - start_t))
    save_images(predictions, args.batch_size, args.save_path)


if __name__ == "__main__":
    args = _parse_args()
    main(args)
