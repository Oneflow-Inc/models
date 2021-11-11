"""
@author: Wang Yizhang <1739601638@qq.com>
"""
import os
import argparse

from model.trainr import CycleGANTrainr


def get_args():
    parser = argparse.ArgumentParser(
        """Train CycleGAN using source dataset and target dataset"""
    )
    parser.add_argument(
        "--logf0s_normalization",
        type=str,
        help="Cached location for log f0s normalized",
        default="./cache/logf0s_normalization.npz",
    )
    parser.add_argument(
        "--mcep_normalization",
        type=str,
        help="Cached location for mcep normalization",
        default="./cache/mcep_normalization.npz",
    )
    parser.add_argument(
        "--coded_sps_A_norm",
        type=str,
        help="mcep norm for data A",
        default="./cache/coded_sps_A_norm.pickle",
    )
    parser.add_argument(
        "--coded_sps_B_norm",
        type=str,
        help="mcep norm for data B",
        default="./cache/coded_sps_B_norm.pickle",
    )
    parser.add_argument(
        "--model_checkpoint",
        type=str,
        help="location where you want to save the model",
        default="./model_checkpoint/",
    )
    parser.add_argument(
        "--resume_training_at",
        type=str,
        help="Location of the pre-trained model to resume training",
        default=None,
    )
    parser.add_argument(
        "--validation_A_dir",
        type=str,
        help="validation set for sound source A",
        default="./data/S0913/",
    )
    parser.add_argument(
        "--output_A_dir",
        type=str,
        help="output for converted Sound Source A",
        default="./converted_sound/S0913",
    )
    parser.add_argument(
        "--validation_B_dir",
        type=str,
        help="Validation set for sound source B",
        default="./data/gaoxiaosong/",
    )
    parser.add_argument(
        "--output_B_dir",
        type=str,
        help="Output for converted sound Source B",
        default="./converted_sound/gaoxiaosong/",
    )
    args = parser.parse_args()
    return args


def train(argv):
    # Check whether following cached files exists
    if not os.path.exists(argv.logf0s_normalization) or not os.path.exists(
        argv.mcep_normalization
    ):
        print(
            "Cached files do not exist, please run the program preprocess_training.py first"
        )

    cycleGAN = CycleGANTrainr(
        logf0s_normalization=argv.logf0s_normalization,
        mcep_normalization=argv.mcep_normalization,
        coded_sps_A_norm=argv.coded_sps_A_norm,
        coded_sps_B_norm=argv.coded_sps_B_norm,
        model_checkpoint=argv.model_checkpoint,
        validation_A_dir=argv.validation_A_dir,
        output_A_dir=argv.output_A_dir,
        validation_B_dir=argv.validation_B_dir,
        output_B_dir=argv.output_B_dir,
        restart_training_at=argv.resume_training_at,
    )
    cycleGAN.train()


if __name__ == "__main__":
    opt = get_args()
    train(opt)
