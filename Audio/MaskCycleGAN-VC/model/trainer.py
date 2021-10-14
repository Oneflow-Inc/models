import os
import pickle

import numpy as np
import soundfile as sf
import librosa
import oneflow as flow

from model.model import Generator, Discriminator
from utils.dataset import VCDataset
import utils.data_utils as preprocess


class MaskCycleGANVCTrainer(object):
    """Trainer for MaskCycleGAN-VC
    """

    def __init__(self, args):
        """
        Args:
            args (Namespace): Program arguments from argparser
        """
        # Store args
        self.num_epochs = args.num_epochs
        self.start_epoch = args.start_epoch
        self.generator_lr = args.generator_lr
        self.discriminator_lr = args.discriminator_lr
        self.decay_after = args.decay_after
        self.mini_batch_size = args.batch_size
        self.cycle_loss_lambda = args.cycle_loss_lambda
        self.identity_loss_lambda = args.identity_loss_lambda
        self.device = args.device
        self.epochs_per_save = args.epochs_per_save
        self.sample_rate = args.sample_rate
        self.validation_A_dir = os.path.join(
            args.origin_data_dir, args.speaker_A_id)
        self.output_A_dir = os.path.join(
            args.output_data_dir, args.speaker_A_id)
        self.validation_B_dir = os.path.join(
            args.origin_data_dir, args.speaker_B_id)
        self.output_B_dir = os.path.join(
            args.output_data_dir, args.speaker_B_id)
        self.infer_data_dir = args.infer_data_dir
        self.pretrain_models = args.pretrain_models

        # Initialize speakerA's dataset
        self.dataset_A = self.loadPickleFile(os.path.join(
            args.preprocessed_data_dir, args.speaker_A_id, f"{args.speaker_A_id}_normalized.pickle"))
        dataset_A_norm_stats = np.load(os.path.join(
            args.preprocessed_data_dir, args.speaker_A_id, f"{args.speaker_A_id}_norm_stat.npz"))
        self.dataset_A_mean = dataset_A_norm_stats['mean']
        self.dataset_A_std = dataset_A_norm_stats['std']

        # Initialize speakerB's dataset
        self.dataset_B = self.loadPickleFile(os.path.join(
            args.preprocessed_data_dir, args.speaker_B_id, f"{args.speaker_B_id}_normalized.pickle"))
        dataset_B_norm_stats = np.load(os.path.join(
            args.preprocessed_data_dir, args.speaker_B_id, f"{args.speaker_B_id}_norm_stat.npz"))
        self.dataset_B_mean = dataset_B_norm_stats['mean']
        self.dataset_B_std = dataset_B_norm_stats['std']

        # Compute lr decay rate
        self.n_samples = len(self.dataset_A)
        print(f'n_samples = {self.n_samples}')
        self.generator_lr_decay = self.generator_lr / \
            float(self.num_epochs * (self.n_samples // self.mini_batch_size))
        self.discriminator_lr_decay = self.discriminator_lr / \
            float(self.num_epochs * (self.n_samples // self.mini_batch_size))
        print(f'generator_lr_decay = {self.generator_lr_decay}')
        print(f'discriminator_lr_decay = {self.discriminator_lr_decay}')

        # Initialize Train Dataloader
        self.num_frames = args.num_frames

        self.dataset = VCDataset(datasetA=self.dataset_A,
                                 datasetB=self.dataset_B,
                                 n_frames=args.num_frames,
                                 max_mask_len=args.max_mask_len)
        self.train_dataloader = flow.utils.data.DataLoader(dataset=self.dataset,
                                                           batch_size=self.mini_batch_size,
                                                           shuffle=True,
                                                           drop_last=False)

        # Initialize Generators and Discriminators
        self.generator_A2B = Generator().to(self.device)
        self.generator_B2A = Generator().to(self.device)
        self.discriminator_A = Discriminator().to(self.device)
        self.discriminator_B = Discriminator().to(self.device)
        # Discriminator to compute 2 step adversarial loss
        self.discriminator_A2 = Discriminator().to(self.device)
        # Discriminator to compute 2 step adversarial loss
        self.discriminator_B2 = Discriminator().to(self.device)

        # Initialize Optimizers
        g_params = list(self.generator_A2B.parameters()) + \
            list(self.generator_B2A.parameters())
        d_params = list(self.discriminator_A.parameters()) + \
            list(self.discriminator_B.parameters()) + \
            list(self.discriminator_A2.parameters()) + \
            list(self.discriminator_B2.parameters())
        self.generator_optimizer = flow.optim.Adam(
            g_params, lr=self.generator_lr, betas=(0.5, 0.999))
        self.discriminator_optimizer = flow.optim.Adam(
            d_params, lr=self.discriminator_lr, betas=(0.5, 0.999))

    def adjust_lr_rate(self, optimizer, generator):
        """Decays learning rate.

        Args:
            optimizer (torch.optim): torch optimizer
            generator (bool): Whether to adjust generator lr.
        """
        if generator:
            self.generator_lr = max(
                0., self.generator_lr - self.generator_lr_decay)
            for param_groups in optimizer.param_groups:
                param_groups['lr'] = self.generator_lr
        else:
            self.discriminator_lr = max(
                0., self.discriminator_lr - self.discriminator_lr_decay)
            for param_groups in optimizer.param_groups:
                param_groups['lr'] = self.discriminator_lr

    def reset_grad(self):
        """Sets gradients of the generators and discriminators to zero before backpropagation.
        """
        self.generator_optimizer.zero_grad()
        self.discriminator_optimizer.zero_grad()

    def loadPickleFile(self, fileName):
        """Loads a Pickle file.

        Args:
            fileName (str): pickle file path

        Returns:
            file object: The loaded pickle file object
        """
        with open(fileName, 'rb') as f:
            return pickle.load(f)

    def train(self):
        """Implements the training loop for MaskCycleGAN-VC
        """
        for epoch in range(self.start_epoch, self.num_epochs + 1):

            for i, (real_A, mask_A, real_B, mask_B) in enumerate(self.train_dataloader):
                num_iterations = (
                    self.n_samples // self.mini_batch_size) * epoch + i
                if num_iterations > 10000:
                    self.identity_loss_lambda = 0
                if num_iterations > self.decay_after:
                    self.adjust_lr_rate(
                        self.generator_optimizer, generator=True)
                    self.adjust_lr_rate(
                        self.generator_optimizer, generator=False)

                real_A = real_A.to(self.device, dtype=flow.float)
                mask_A = mask_A.to(self.device, dtype=flow.float)
                real_B = real_B.to(self.device, dtype=flow.float)
                mask_B = mask_B.to(self.device, dtype=flow.float)

                # Train Generator
                self.generator_A2B.train()
                self.generator_B2A.train()
                self.discriminator_A.eval()
                self.discriminator_B.eval()
                self.discriminator_A2.eval()
                self.discriminator_B2.eval()

                # Generator Feed Forward
                fake_B = self.generator_A2B(real_A, mask_A)
                cycle_A = self.generator_B2A(fake_B, flow.ones_like(fake_B))
                fake_A = self.generator_B2A(real_B, mask_B)
                cycle_B = self.generator_A2B(fake_A, flow.ones_like(fake_A))
                identity_A = self.generator_B2A(
                    real_A, flow.ones_like(real_A))
                identity_B = self.generator_A2B(
                    real_B, flow.ones_like(real_B))
                d_fake_A = self.discriminator_A(fake_A)
                d_fake_B = self.discriminator_B(fake_B)

                # For Two Step Adverserial Loss
                d_fake_cycle_A = self.discriminator_A2(cycle_A)
                d_fake_cycle_B = self.discriminator_B2(cycle_B)

                # Generator Cycle Loss
                cycleLoss = flow.mean(
                    flow.abs(real_A - cycle_A)) + flow.mean(flow.abs(real_B - cycle_B))

                # Generator Identity Loss
                identityLoss = flow.mean(
                    flow.abs(real_A - identity_A)) + flow.mean(flow.abs(real_B - identity_B))

                # Generator Loss
                g_loss_A2B = flow.mean((1 - d_fake_B) ** 2)
                g_loss_B2A = flow.mean((1 - d_fake_A) ** 2)

                # Generator Two Step Adverserial Loss
                generator_loss_A2B_2nd = flow.mean((1 - d_fake_cycle_B) ** 2)
                generator_loss_B2A_2nd = flow.mean((1 - d_fake_cycle_A) ** 2)

                # Total Generator Loss
                g_loss = g_loss_A2B + g_loss_B2A + \
                    generator_loss_A2B_2nd + generator_loss_B2A_2nd + \
                    self.cycle_loss_lambda * cycleLoss + self.identity_loss_lambda * identityLoss

                # Backprop for Generator
                self.reset_grad()
                g_loss.backward()
                self.generator_optimizer.step()

                # Train Discriminator
                self.generator_A2B.eval()
                self.generator_B2A.eval()
                self.discriminator_A.train()
                self.discriminator_B.train()
                self.discriminator_A2.train()
                self.discriminator_B2.train()

                # Discriminator Feed Forward
                d_real_A = self.discriminator_A(real_A)
                d_real_B = self.discriminator_B(real_B)
                d_real_A2 = self.discriminator_A2(real_A)
                d_real_B2 = self.discriminator_B2(real_B)
                generated_A = self.generator_B2A(real_B, mask_B)
                d_fake_A = self.discriminator_A(generated_A)

                # For Two Step Adverserial Loss A->B
                cycled_B = self.generator_A2B(
                    generated_A, flow.ones_like(generated_A))
                d_cycled_B = self.discriminator_B2(cycled_B)

                generated_B = self.generator_A2B(real_A, mask_A)
                d_fake_B = self.discriminator_B(generated_B)

                # For Two Step Adverserial Loss B->A
                cycled_A = self.generator_B2A(
                    generated_B, flow.ones_like(generated_B))
                d_cycled_A = self.discriminator_A2(cycled_A)

                # Loss Functions
                d_loss_A_real = flow.mean((1 - d_real_A) ** 2)
                d_loss_A_fake = flow.mean((0 - d_fake_A) ** 2)
                d_loss_A = (d_loss_A_real + d_loss_A_fake) / 2.0

                d_loss_B_real = flow.mean((1 - d_real_B) ** 2)
                d_loss_B_fake = flow.mean((0 - d_fake_B) ** 2)
                d_loss_B = (d_loss_B_real + d_loss_B_fake) / 2.0

                # Two Step Adverserial Loss
                d_loss_A_cycled = flow.mean((0 - d_cycled_A) ** 2)
                d_loss_B_cycled = flow.mean((0 - d_cycled_B) ** 2)
                d_loss_A2_real = flow.mean((1 - d_real_A2) ** 2)
                d_loss_B2_real = flow.mean((1 - d_real_B2) ** 2)
                d_loss_A_2nd = (d_loss_A2_real + d_loss_A_cycled) / 2.0
                d_loss_B_2nd = (d_loss_B2_real + d_loss_B_cycled) / 2.0

                # Final Loss for discriminator with the Two Step Adverserial Loss
                d_loss = (d_loss_A + d_loss_B) / 2.0 + \
                    (d_loss_A_2nd + d_loss_B_2nd) / 2.0

                # Backprop for Discriminator
                self.reset_grad()
                d_loss.backward()
                self.discriminator_optimizer.step()

                if (i + 1) % 2 == 0:
                    print(
                        "Iter:{} Generator Loss:{:.4f} Discrimator Loss:{:.4f} GA2B:{:.4f} GB2A:{:.4f} G_id:{:.4f} G_cyc:{:.4f} D_A:{:.4f} D_B:{:.4f}".format(
                            num_iterations,
                            g_loss.item(),
                            d_loss.item(), g_loss_A2B, g_loss_B2A, identityLoss, cycleLoss, d_loss_A,
                            d_loss_B))

            # Save each model checkpoint and validation
            if epoch % self.epochs_per_save == 0 and epoch != 0:
                self.saveModelCheckPoint(epoch, PATH='model_checkpoint')
                self.validation_for_A_dir()
                self.validation_for_B_dir()

    def infer(self):
        """Implements the infering loop for MaskCycleGAN-VC
        """
        # load pretrain models
        self.loadModel(self.pretrain_models)

        num_mcep = 80
        sampling_rate = self.sample_rate
        frame_period = 5.0
        infer_A_dir = self.infer_data_dir

        print("Generating Validation Data B from A...")
        for file in os.listdir(infer_A_dir):
            filePath = os.path.join(infer_A_dir, file)
            wav, _ = librosa.load(filePath, sr=sampling_rate, mono=True)
            wav = preprocess.wav_padding(wav=wav,
                                         sr=sampling_rate,
                                         frame_period=frame_period,
                                         multiple=4)
            f0, timeaxis, sp, ap = preprocess.world_decompose(
                wav=wav, fs=sampling_rate, frame_period=frame_period)
            f0_converted = preprocess.pitch_conversion(f0=f0,
                                                       mean_log_src=self.dataset_A_mean,
                                                       std_log_src=self.dataset_A_std,
                                                       mean_log_target=self.dataset_B_mean,
                                                       std_log_target=self.dataset_B_std)
            coded_sp = preprocess.world_encode_spectral_envelop(
                sp=sp, fs=sampling_rate, dim=num_mcep)
            coded_sp_transposed = coded_sp.T
            coded_sp_norm = (coded_sp_transposed -
                             self.dataset_A_mean) / self.dataset_A_std
            coded_sp_norm = np.array([coded_sp_norm])

            if flow.cuda.is_available():
                coded_sp_norm = flow.tensor(coded_sp_norm).cuda().float()
            else:
                coded_sp_norm = flow.tensor(coded_sp_norm).float()

            coded_sp_converted_norm = self.generator_A2B(
                coded_sp_norm, flow.ones_like(coded_sp_norm))
            coded_sp_converted_norm = coded_sp_converted_norm.cpu().detach().numpy()
            coded_sp_converted_norm = np.squeeze(coded_sp_converted_norm)
            coded_sp_converted = coded_sp_converted_norm * \
                self.dataset_B_std + self.dataset_B_mean
            coded_sp_converted = coded_sp_converted.T
            coded_sp_converted = np.ascontiguousarray(
                coded_sp_converted).astype(np.double)
            decoded_sp_converted = preprocess.world_decode_spectral_envelop(
                coded_sp=coded_sp_converted, fs=sampling_rate)

            wav_transformed = preprocess.world_speech_synthesis(f0=f0_converted[0],
                                                                decoded_sp=decoded_sp_converted,
                                                                ap=ap,
                                                                fs=sampling_rate,
                                                                frame_period=frame_period)

            sf.write(os.path.join(infer_A_dir, 'convert_' + os.path.basename(file)),
                     wav_transformed,
                     sampling_rate)

    def validation_for_A_dir(self):
        num_mcep = 80
        sampling_rate = 22050
        frame_period = 5.0
        validation_A_dir = self.validation_A_dir
        output_A_dir = self.output_A_dir

        os.makedirs(output_A_dir, exist_ok=True)

        print("Generating Validation Data B from A...")
        for file in os.listdir(validation_A_dir):
            filePath = os.path.join(validation_A_dir, file)
            wav, _ = librosa.load(filePath, sr=sampling_rate, mono=True)
            wav = preprocess.wav_padding(wav=wav,
                                         sr=sampling_rate,
                                         frame_period=frame_period,
                                         multiple=4)
            f0, timeaxis, sp, ap = preprocess.world_decompose(
                wav=wav, fs=sampling_rate, frame_period=frame_period)
            f0_converted = preprocess.pitch_conversion(f0=f0,
                                                       mean_log_src=self.dataset_A_mean,
                                                       std_log_src=self.dataset_A_std,
                                                       mean_log_target=self.dataset_B_mean,
                                                       std_log_target=self.dataset_B_std)
            coded_sp = preprocess.world_encode_spectral_envelop(
                sp=sp, fs=sampling_rate, dim=num_mcep)
            coded_sp_transposed = coded_sp.T
            coded_sp_norm = (coded_sp_transposed -
                             self.dataset_A_mean) / self.dataset_A_std
            coded_sp_norm = np.array([coded_sp_norm])

            if flow.cuda.is_available():
                coded_sp_norm = flow.tensor(coded_sp_norm).cuda().float()
            else:
                coded_sp_norm = flow.tensor(coded_sp_norm).float()

            coded_sp_converted_norm = self.generator_A2B(
                coded_sp_norm, flow.ones_like(coded_sp_norm))
            coded_sp_converted_norm = coded_sp_converted_norm.cpu().detach().numpy()
            coded_sp_converted_norm = np.squeeze(coded_sp_converted_norm)
            coded_sp_converted = coded_sp_converted_norm * \
                self.dataset_B_std + self.dataset_B_mean
            coded_sp_converted = coded_sp_converted.T
            coded_sp_converted = np.ascontiguousarray(
                coded_sp_converted).astype(np.double)
            decoded_sp_converted = preprocess.world_decode_spectral_envelop(
                coded_sp=coded_sp_converted, fs=sampling_rate)

            wav_transformed = preprocess.world_speech_synthesis(f0=f0_converted[0],
                                                                decoded_sp=decoded_sp_converted,
                                                                ap=ap,
                                                                fs=sampling_rate,
                                                                frame_period=frame_period)

            sf.write(os.path.join(output_A_dir, 'convert_' + os.path.basename(file)),
                     wav_transformed,
                     sampling_rate)

    def validation_for_B_dir(self):
        num_mcep = 80
        sampling_rate = 22050
        frame_period = 5.0
        validation_B_dir = self.validation_B_dir
        output_B_dir = self.output_B_dir

        os.makedirs(output_B_dir, exist_ok=True)

        print("Generating Validation Data A from B...")
        for file in os.listdir(validation_B_dir):
            filePath = os.path.join(validation_B_dir, file)
            wav, _ = librosa.load(filePath, sr=sampling_rate, mono=True)
            wav = preprocess.wav_padding(wav=wav,
                                         sr=sampling_rate,
                                         frame_period=frame_period,
                                         multiple=4)
            f0, timeaxis, sp, ap = preprocess.world_decompose(
                wav=wav, fs=sampling_rate, frame_period=frame_period)
            f0_converted = preprocess.pitch_conversion(f0=f0,
                                                       mean_log_src=self.dataset_B_mean,
                                                       std_log_src=self.dataset_B_std,
                                                       mean_log_target=self.dataset_A_mean,
                                                       std_log_target=self.dataset_A_std)
            coded_sp = preprocess.world_encode_spectral_envelop(
                sp=sp, fs=sampling_rate, dim=num_mcep)
            coded_sp_transposed = coded_sp.T
            coded_sp_norm = (coded_sp_transposed -
                             self.dataset_B_mean) / self.dataset_B_std
            coded_sp_norm = np.array([coded_sp_norm])

            if flow.cuda.is_available():
                coded_sp_norm = flow.tensor(coded_sp_norm).cuda().float()
            else:
                coded_sp_norm = flow.tensor(coded_sp_norm).float()

            coded_sp_converted_norm = self.generator_B2A(
                coded_sp_norm, flow.ones_like(coded_sp_norm))
            coded_sp_converted_norm = coded_sp_converted_norm.cpu().detach().numpy()
            coded_sp_converted_norm = np.squeeze(coded_sp_converted_norm)
            coded_sp_converted = coded_sp_converted_norm * \
                self.dataset_A_std + self.dataset_A_mean
            coded_sp_converted = coded_sp_converted.T
            coded_sp_converted = np.ascontiguousarray(
                coded_sp_converted).astype(np.double)
            decoded_sp_converted = preprocess.world_decode_spectral_envelop(
                coded_sp=coded_sp_converted, fs=sampling_rate)

            wav_transformed = preprocess.world_speech_synthesis(f0=f0_converted[0],
                                                                decoded_sp=decoded_sp_converted,
                                                                ap=ap,
                                                                fs=sampling_rate,
                                                                frame_period=frame_period)

            sf.write(os.path.join(output_B_dir, 'convert_' + os.path.basename(file)),
                     wav_transformed,
                     sampling_rate)

    def saveModelCheckPoint(self, epoch, PATH):
        flow.save(self.generator_A2B.state_dict(),
                  os.path.join(PATH, "generator_A2B_%d" % epoch))
        flow.save(self.generator_B2A.state_dict(),
                  os.path.join(PATH, "generator_B2A_%d" % epoch))
        flow.save(self.discriminator_A.state_dict(),
                  os.path.join(PATH, "discriminator_A_%d" % epoch))
        flow.save(self.discriminator_B.state_dict(),
                  os.path.join(PATH, "discriminator_B_%d" % epoch))

    def loadModel(self, PATH):
        self.generator_A2B.load_state_dict(
            flow.load(os.path.join(PATH, "generator_A2B")))
        self.generator_B2A.load_state_dict(
            flow.load(os.path.join(PATH, "generator_B2A")))
        self.discriminator_A.load_state_dict(
            flow.load(os.path.join(PATH, "discriminator_A")))
        self.discriminator_B.load_state_dict(
            flow.load(os.path.join(PATH, "discriminator_B")))
