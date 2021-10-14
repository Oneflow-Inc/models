import os
import random
import argparse

import numpy as np


class CycleGANArgParser(object):
    def __init__(self):
        self.parser = argparse.ArgumentParser(description='args')

        self.parser.add_argument(
            '--batch_size', type=int, default=9, help='Batch size.')
        self.parser.add_argument(
            '--seed', type=int, default=0, help='Random Seed.')
        self.parser.add_argument('--device', type=str, default='cuda')

        self.parser.add_argument('--epochs_per_save', type=int, default=1,
                                 help='Number of epochs between saving the model.')
        self.parser.add_argument(
            '--start_epoch', type=int, default=1, help='Epoch to start training')
        self.parser.add_argument(
            '--num_epochs', type=int, default=6500, help='Number of epochs to train.')
        self.parser.add_argument(
            '--decay_after', type=float, default=2e5, help='Decay learning rate after n iterations.')

        self.parser.add_argument(
            '--sample_rate', type=int, default=22050, help='Sampling rate of mel-spectrograms.')
        self.parser.add_argument(
            '--speaker_A_id', type=str, default="VCC2SF3", help='Source speaker id.')
        self.parser.add_argument(
            '--speaker_B_id', type=str, default="VCC2SM3", help='Target speaker id.')
        self.parser.add_argument(
            '--origin_data_dir', type=str, default="vcc2018/vcc2018_training/", help='Directory containing origin dataset files.')
        self.parser.add_argument(
            '--preprocessed_data_dir', type=str, default="vcc2018_preprocessed/vcc2018_training/", help='Directory containing preprocessed dataset files.')
        self.parser.add_argument(
            '--pretrain_models', type=str, default="pretrain_models/", help='Directory containing pretrain models.')
        self.parser.add_argument(
            '--infer_data_dir', type=str, default="sample/", help='Directory containing infer dataset files.')
        self.parser.add_argument(
            '--output_data_dir', type=str, default="./converted_sound/", help='Directory containing output dataset files.')

        self.parser.add_argument(
            '--generator_lr', type=float, default=2e-4, help='Initial generator learning rate.')
        self.parser.add_argument(
            '--discriminator_lr', type=float, default=1e-4, help='Initial discrminator learning rate.')

        self.parser.add_argument(
            '--cycle_loss_lambda', type=float, default=10, help='Lambda value for cycle consistency loss.')
        self.parser.add_argument(
            '--identity_loss_lambda', type=float, default=5, help='Lambda value for identity loss.')

        self.parser.add_argument(
            '--num_frames', type=int, default=64, help='Num frames per training sample.'
        )
        self.parser.add_argument(
            '--max_mask_len', type=int, default=32, help='Maximum length of mask for Mask-CycleGAN-VC.'
        )

        self.parser.set_defaults(
            batch_size=9, num_epochs=50, decay_after=1e4, start_epoch=1, num_frames=64)

    def parse_args(self):
        args = self.parser.parse_args()

        # Limit sources of nondeterministic behavior
        os.environ['PYTHONHASHSEED'] = str(args.seed)
        random.seed(args.seed)
        np.random.seed(args.seed)

        self.print_options(args)

        return args

    def print_options(self, args):
        """
        Function that prints current options

        Parameters
        ----------
        args : Namespace
            Arguments for models and model testing
        """
        message = ''
        message += '----------------- Options ---------------\n'
        for k, v in sorted(vars(args).items()):
            message += '{:>25}: {:<30}\n'.format(str(k), str(v))
        message += '----------------- End -------------------'
        print(message)
