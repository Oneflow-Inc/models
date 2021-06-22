import oneflow.experimental as flow
import argparse
import numpy as np
from utils.dataset import load_facades
from utils.utils import to_tensor, save_images

def main(args):
    test_x, test_y = load_facades(mode="test")
    # run every epoch to shuffle
    ind = np.random.choice(len(test_x) // args.batch_size)
    test_inp = to_tensor(test_x[ind * args.batch_size: (
        ind + 1) * args.batch_size].astype(np.float32)).to("cuda")
    test_target = to_tensor(test_y[ind * args.batch_size: (
        ind + 1) * args.batch_size].astype(np.float32)).to("cuda")

    gout, test_image_error = args.eval_generator(
        test_inp, test_target)
    # save images
    # self.save_images(g_out, inp, target, epoch_idx, name="train")
    save_images(gout, test_inp, test_target,
                        epoch_idx, name="test")
    print("############## evaluation ###############")
    print("{}th epoch, {}th batch, test_image_error:{}".format(
        epoch_idx + 1, batch_idx + 1, test_image_error.mean()))

if __name__ == "__main__":
    flow.enable_eager_execution()
    parser = argparse.ArgumentParser(description="oneflow PIX2PIX")
    parser.add_argument("--path", type=str, default='./', required=False)
    parser.add_argument("--load", type=str, default="", required=False,
                        help="the path to continue training the model")
    parser.add_argument("--batch_size", type=int, default=32, required=False)
    args = parser.parse_args()
    main(args)