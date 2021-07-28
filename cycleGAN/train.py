"""
The code refers to DeepLearningForFun(https://github.com/Ldpe2G/DeepLearningForFun/tree/master/Oneflow-Python/CycleGAN) by Ldpe2G
and pytorch-CycleGAN-and-pix2pix(https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix) by junyuanz for implementation.
"""
import argparse

from cycleGAN import CycleGANModel
from image import load_image2ndarray
import os
import random
import oneflow as flow

def main(args):
    datasetA = os.listdir(args.datasetA_path)
    datasetB = os.listdir(args.datasetB_path)

    datasetA_num = len(datasetA)
    datasetB_num = len(datasetB)
    print("dataset A size: %d" % datasetA_num)
    print("dataset B size: %d" % datasetB_num) 

    train_iters = min(datasetA_num, datasetB_num)          

    model = CycleGANModel(args)

    for e in range(args.train_epoch):    # outer loop for different epochs; we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>
        random.shuffle(datasetA)
        random.shuffle(datasetB)
        model.update_learning_rate()
        for i in range(train_iters):
            model.set_input(load_image2ndarray(args.datasetA_path + datasetA[i]), \
                load_image2ndarray(args.datasetB_path + datasetB[i]))
            model.optimize_parameters()

            if (i + 1) % 100 == 0:
                model.log_loss(e, i + 1)
                model.save_result(args.save_tmp_image_path + "epoch_%d_iter_%d.jpg" % (e, i + 1))
                
        if e % 10 == 0:
            flow.save(model.netG_A.state_dict(), args.checkpoint_save_dir + args.dataset_name + "_A2B_checkpoint_epoch_%d_%f_%f" % (e, args.beta1, args.beta2))
            flow.save(model.netG_B.state_dict(), args.checkpoint_save_dir + args.dataset_name + "_B2A_checkpoint_epoch_%d_%f_%f" % (e, args.beta1, args.beta2))
            flow.save(model.netD_A.state_dict(), args.checkpoint_save_dir + args.dataset_name + "_D_A_checkpoint_epoch_%d_%f_%f" % (e, args.beta1, args.beta2))
            flow.save(model.netD_B.state_dict(), args.checkpoint_save_dir + args.dataset_name + "_D_B_checkpoint_epoch_%d_%f_%f" % (e, args.beta1, args.beta2))



def get_parser(parser = None):
    parser = argparse.ArgumentParser("flags for cycle gan")

    parser.add_argument("--dataset_name", type = str, default = "", help = "dataset name")

    parser.add_argument("--datasetA_path", type = str, default = "", help = "dataset A path")
    parser.add_argument("--datasetB_path", type = str, default = "", help = "dataset B path")

    # image preprocess
    parser.add_argument("--crop_size", type = int, default = 286)
    parser.add_argument("--load_size", type = int, default = 256)
    parser.add_argument("--resize_and_crop", type = bool, default = True)

    # checkpoint
    parser.add_argument("--checkpoint_load_epoch", type = int, default = None, help = "load previous saved checkpoint from")
    parser.add_argument("--checkpoint_save_dir", type = str, default = "checkpoints/", help = "save checkpoint to")
    parser.add_argument("--save_tmp_image_path", type = str, default = 'train_temp_image.jpg', help = "image path")

    # hyper-parameters
    parser.add_argument("--train_epoch", type = int, default = 300)
    parser.add_argument("--learning_rate", type = float, default = 0.0002)
    parser.add_argument("--n_blocks", type=int, default = 9, help = "res blocks of generator")
    parser.add_argument('--lambda_A', type=float, default = 10.0, help = 'weight for cycle loss (A -> B -> A)')
    parser.add_argument('--lambda_B', type=float, default = 10.0, help = 'weight for cycle loss (B -> A -> B)')
    parser.add_argument('--lambda_identity', type=float, default = 0.5, help='use identity mapping. Setting lambda_identity other than 0 has an effect of scaling the weight of the identity mapping loss. For example, if the weight of the identity loss should be 10 times smaller than the weight of the reconstruction loss, please set lambda_identity = 0.1')
    parser.add_argument('--ngf', type = int, default = 64, help = '# of gen filters in the last conv layer')
    parser.add_argument('--ndf', type = int, default = 64, help = '# of discrim filters in the first conv layer')
    parser.add_argument('--device', type = str, default = "cuda", help = "specify device: cpu or cuda")
    parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate for adam')
    parser.add_argument('--pool_size', type=int, default=50, help='the size of image buffer that stores previously generated images')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam')
    parser.add_argument('--beta2', type=float, default=0.5, help='beta1 for adam')

    return parser

if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    main(args)
