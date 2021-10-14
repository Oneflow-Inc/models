"""
@author: Wang Yizhang <1739601638@qq.com>
"""
from args.cycleGAN_arg_parser import CycleGANArgParser
from model.trainer import MaskCycleGANVCTrainer


if __name__ == "__main__":
    parser = CycleGANArgParser()
    args = parser.parse_args()
    cycleGAN = MaskCycleGANVCTrainer(args)
    cycleGAN.infer()
