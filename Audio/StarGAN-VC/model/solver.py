import os
from datetime import datetime
import random

from sklearn.preprocessing import LabelBinarizer
from pyworld import decode_spectral_envelope, synthesize
import soundfile as sf
import ast
import numpy as np
import oneflow as flow
import oneflow.nn as nn

from model.data_loader import TestSet
from model.model import Discriminator, DomainClassifier, Generator
from utils.utility import Normalizer, speakers
from utils.preprocess import FRAMES, SAMPLE_RATE, FFTSIZE


class Solver(object):
    def __init__(self, data_loader, config):

        self.config = config
        self.data_loader = data_loader

        # Model configurations.
        self.lambda_cycle = config.lambda_cycle
        self.lambda_cls = config.lambda_cls
        self.lambda_identity = config.lambda_identity

        # Training configurations.
        self.data_dir = config.data_dir
        self.test_dir = config.test_dir
        self.batch_size = config.batch_size
        self.num_iters = config.num_iters
        self.num_iters_decay = config.num_iters_decay
        self.g_lr = config.g_lr
        self.d_lr = config.d_lr
        self.c_lr = config.c_lr
        self.n_critic = config.n_critic
        self.beta1 = config.beta1
        self.beta2 = config.beta2
        self.resume_iters = config.resume_iters

        # Test configurations.
        self.pretrain_models = config.pretrain_models
        self.sample_dir = config.sample_dir
        self.trg_speaker = ast.literal_eval(config.trg_speaker)
        self.src_speaker = config.src_speaker

        # Miscellaneous.
        self.device = flow.device("cuda:0" if flow.cuda.is_available() else "cpu")
        self.spk_enc = LabelBinarizer().fit(speakers)

        # Directories.
        self.model_save_dir = config.model_save_dir
        self.result_dir = config.result_dir
        self.use_gradient_penalty = config.use_gradient_penalty

        # Step size.
        self.log_step = config.log_step
        self.sample_step = config.sample_step
        self.model_save_step = config.model_save_step
        self.lr_update_step = config.lr_update_step

        # Build the model.
        self.build_model()

    def build_model(self):
        self.G = Generator()
        self.D = Discriminator()
        self.C = DomainClassifier()

        self.g_optimizer = flow.optim.Adam(
            self.G.parameters(), self.g_lr, [self.beta1, self.beta2]
        )
        self.d_optimizer = flow.optim.Adam(
            self.D.parameters(), self.d_lr, [self.beta1, self.beta2]
        )
        self.c_optimizer = flow.optim.Adam(
            self.C.parameters(), self.c_lr, [self.beta1, self.beta2]
        )

        self.print_network(self.G, "G")
        self.print_network(self.D, "D")
        self.print_network(self.C, "C")

        self.G.to(self.device)
        self.D.to(self.device)
        self.C.to(self.device)

    def print_network(self, model, name):
        """Print out the network information."""
        num_params = 0
        for p in model.parameters():
            num_params += p.numel()
        print(model)
        print(name)
        print("The number of parameters: {}".format(num_params))

    def update_lr(self, g_lr, d_lr, c_lr):
        """Decay learning rates of the generator and discriminator and classifier."""
        for param_group in self.g_optimizer.param_groups:
            param_group["lr"] = g_lr
        for param_group in self.d_optimizer.param_groups:
            param_group["lr"] = d_lr
        for param_group in self.c_optimizer.param_groups:
            param_group["lr"] = c_lr

    def train(self):
        # Learning rate cache for decaying.
        g_lr = self.g_lr
        d_lr = self.d_lr
        c_lr = self.c_lr

        start_iters = 0
        if self.resume_iters:
            pass

        norm = Normalizer()
        data_iter = iter(self.data_loader)

        print("Start training......")
        start_time = datetime.now()

        for i in range(start_iters, self.num_iters):
            # Preprocess input data
            # Fetch real images and labels.
            try:
                x_real, speaker_idx_org, label_org = next(data_iter)
            except:
                data_iter = iter(self.data_loader)
                x_real, speaker_idx_org, label_org = next(data_iter)

            # Generate target domain labels randomly.
            rand_idx = flow.randperm(label_org.size(0))
            label_trg = label_org[rand_idx]
            speaker_idx_trg = speaker_idx_org[rand_idx]

            x_real = x_real.to(self.device)
            # Original domain one-hot labels.
            label_org = label_org.to(self.device)
            # Target domain one-hot labels.
            label_trg = label_trg.to(self.device)
            speaker_idx_org = speaker_idx_org.to(self.device)
            speaker_idx_trg = speaker_idx_trg.to(self.device)

            # Train the discriminator
            # Compute loss with real audio frame.
            CELoss = nn.CrossEntropyLoss()
            cls_real = self.C(x_real)
            cls_loss_real = CELoss(input=cls_real, target=speaker_idx_org)

            self.reset_grad()
            cls_loss_real.backward()
            self.c_optimizer.step()
            # Logging.
            loss = {}
            loss["C/C_loss"] = cls_loss_real.item()

            out_r = self.D(x_real, label_org)
            # Compute loss with fake audio frame.
            x_fake = self.G(x_real, label_trg)
            out_f = self.D(x_fake.detach(), label_trg)
            d_loss_t = nn.BCEWithLogitsLoss()(
                input=out_f, target=flow.zeros_like(out_f).float()
            ) + nn.BCEWithLogitsLoss()(
                input=out_r, target=flow.ones_like(out_r).float()
            )

            out_cls = self.C(x_fake)
            d_loss_cls = CELoss(input=out_cls, target=speaker_idx_trg)

            # Compute loss for gradient penalty.
            alpha = flow.rand(x_real.size(0), 1, 1, 1).to(self.device)
            x_hat = (
                (alpha * x_real + (1 - alpha) * x_fake).detach().requires_grad_(True)
            )
            out_src = self.D(x_hat, label_trg)

            # TODO: Second-order derivation is not currently supported in oneflow, so gradient penalty cannot be used temporarily.
            if self.use_gradient_penalty:
                d_loss_gp = self.gradient_penalty(out_src, x_hat)
                d_loss = d_loss_t + self.lambda_cls * d_loss_cls + 5 * d_loss_gp
            else:
                d_loss = d_loss_t + self.lambda_cls * d_loss_cls

            self.reset_grad()
            d_loss.backward()
            self.d_optimizer.step()

            loss["D/D_loss"] = d_loss.item()

            # Train the generator
            if (i + 1) % self.n_critic == 0:
                # Original-to-target domain.
                x_fake = self.G(x_real, label_trg)
                g_out_src = self.D(x_fake, label_trg)
                g_loss_fake = nn.BCEWithLogitsLoss()(
                    input=g_out_src, target=flow.ones_like(g_out_src).float()
                )

                out_cls = self.C(x_real)
                g_loss_cls = CELoss(input=out_cls, target=speaker_idx_org)

                # Target-to-original domain.
                x_reconst = self.G(x_fake, label_org)
                g_loss_rec = nn.L1Loss()(x_reconst, x_real)

                # Original-to-Original domain(identity).
                x_fake_iden = self.G(x_real, label_org)
                id_loss = nn.L1Loss()(x_fake_iden, x_real)

                # Backward and optimize.
                g_loss = (
                    g_loss_fake
                    + self.lambda_cycle * g_loss_rec
                    + self.lambda_cls * g_loss_cls
                    + self.lambda_identity * id_loss
                )

                self.reset_grad()
                g_loss.backward()
                self.g_optimizer.step()

                # Logging.
                loss["G/loss_fake"] = g_loss_fake.item()
                loss["G/loss_rec"] = g_loss_rec.item()
                loss["G/loss_cls"] = g_loss_cls.item()
                loss["G/loss_id"] = id_loss.item()
                loss["G/g_loss"] = g_loss.item()

            # Miscellaneous
            # Print out training information.
            if (i + 1) % self.log_step == 0:
                et = datetime.now() - start_time
                et = str(et)[:-7]
                log = "Elapsed [{}], Iteration [{}/{}]".format(
                    et, i + 1, self.num_iters
                )
                for tag, value in loss.items():
                    log += ", {}: {:.4f}".format(tag, value)
                print(log)

            # Translate fixed images for debugging.
            if (i + 1) % self.sample_step == 0:
                with flow.no_grad():
                    d, speaker = TestSet(self.test_dir).test_data()
                    target = random.choice([x for x in speakers if x != speaker])
                    label_t = self.spk_enc.transform([target])[0]
                    label_t = np.asarray([label_t])

                    for filename, content in d.items():
                        f0 = content["f0"]
                        ap = content["ap"]
                        sp_norm_pad = self.pad_coded_sp(content["coded_sp_norm"])

                        convert_result = []
                        for start_idx in range(
                            0, sp_norm_pad.shape[1] - FRAMES + 1, FRAMES
                        ):
                            one_seg = sp_norm_pad[:, start_idx : start_idx + FRAMES]

                            one_seg = flow.Tensor(one_seg).to(self.device)
                            one_seg = one_seg.view(
                                1, 1, one_seg.size(0), one_seg.size(1)
                            )
                            l = flow.Tensor(label_t)
                            one_seg = one_seg.to(self.device)
                            l = l.to(self.device)
                            one_set_return = self.G(one_seg, l).detach().cpu().numpy()
                            one_set_return = np.squeeze(one_set_return)
                            one_set_return = norm.backward_process(
                                one_set_return, target
                            )
                            convert_result.append(one_set_return)

                        convert_con = np.concatenate(convert_result, axis=1)
                        convert_con = convert_con[
                            :, 0 : content["coded_sp_norm"].shape[1]
                        ]
                        contigu = np.ascontiguousarray(convert_con.T, dtype=np.float64)
                        decoded_sp = decode_spectral_envelope(
                            contigu, SAMPLE_RATE, fft_size=FFTSIZE
                        )
                        f0_converted = norm.pitch_conversion(f0, speaker, target)
                        wav = synthesize(f0_converted, decoded_sp, ap, SAMPLE_RATE)

                        name = f"{speaker}-{target}_iter{i+1}_{filename}"
                        path = os.path.join(self.sample_dir, name)
                        print(f"[save]:{path}")
                        sf.write(path, wav, SAMPLE_RATE)

            # Save model checkpoints.
            if (i + 1) % self.model_save_step == 0:
                G_path = os.path.join(self.model_save_dir, "{}-G".format(i + 1))
                D_path = os.path.join(self.model_save_dir, "{}-D".format(i + 1))
                C_path = os.path.join(self.model_save_dir, "{}-C".format(i + 1))
                flow.save(self.G.state_dict(), G_path)
                flow.save(self.D.state_dict(), D_path)
                flow.save(self.C.state_dict(), C_path)
                print("Saved model checkpoints into {}...".format(self.model_save_dir))

            # Decay learning rates.
            if (i + 1) % self.lr_update_step == 0 and (i + 1) > (
                self.num_iters - self.num_iters_decay
            ):
                g_lr -= self.g_lr / float(self.num_iters_decay)
                d_lr -= self.d_lr / float(self.num_iters_decay)
                c_lr -= self.c_lr / float(self.num_iters_decay)
                self.update_lr(g_lr, d_lr, c_lr)
                print("Decayed learning rates, g_lr: {}, d_lr: {}.".format(g_lr, d_lr))

    def gradient_penalty(self, y, x):
        """Compute gradient penalty: (L2_norm(dy/dx) - 1)**2."""
        weight = flow.ones(y.size()).to(self.device)

        dydx = flow.autograd.grad(
            outputs=y, inputs=x, out_grads=weight, retain_graph=True, create_graph=True
        )[0]

        dydx = dydx.view(dydx.size(0), -1)
        dydx_l2norm = flow.sqrt(flow.sum(dydx ** 2, dim=1))

        return flow.mean((dydx_l2norm - 1) ** 2)

    def reset_grad(self):
        """Reset the gradient buffers."""
        self.g_optimizer.zero_grad()
        self.d_optimizer.zero_grad()
        self.c_optimizer.zero_grad()

    def restore_model(self, model_save_dir):
        """Restore the tra,zined generator and discriminator."""
        print("Loading the pretrain models...")
        G_path = os.path.join(model_save_dir, "200000-G")
        D_path = os.path.join(model_save_dir, "200000-D")
        C_path = os.path.join(model_save_dir, "200000-C")
        self.G.load_state_dict(flow.load(G_path))
        self.D.load_state_dict(flow.load(D_path))
        self.C.load_state_dict(flow.load(C_path))

    @staticmethod
    def pad_coded_sp(coded_sp_norm):
        f_len = coded_sp_norm.shape[1]
        if f_len >= FRAMES:
            pad_length = FRAMES - (f_len - (f_len // FRAMES) * FRAMES)
        elif f_len < FRAMES:
            pad_length = FRAMES - f_len

        sp_norm_pad = np.hstack(
            (coded_sp_norm, np.zeros((coded_sp_norm.shape[0], pad_length)))
        )
        return sp_norm_pad

    def test(self):
        """Translate speech using StarGAN ."""
        # Load the trained generator.
        self.restore_model(self.pretrain_models)
        norm = Normalizer()

        # Set data loader.
        d, speaker = TestSet(self.test_dir).test_data(self.src_speaker)
        targets = self.trg_speaker

        for target in targets:
            print(target)
            assert target in speakers
            label_t = self.spk_enc.transform([target])[0]
            label_t = np.asarray([label_t])

            with flow.no_grad():

                for filename, content in d.items():
                    f0 = content["f0"]
                    ap = content["ap"]
                    sp_norm_pad = self.pad_coded_sp(content["coded_sp_norm"])

                    convert_result = []
                    for start_idx in range(
                        0, sp_norm_pad.shape[1] - FRAMES + 1, FRAMES
                    ):
                        one_seg = sp_norm_pad[:, start_idx : start_idx + FRAMES]

                        one_seg = flow.Tensor(one_seg).to(self.device)
                        one_seg = one_seg.view(1, 1, one_seg.size(0), one_seg.size(1))
                        l = flow.Tensor(label_t)
                        one_seg = one_seg.to(self.device)
                        l = l.to(self.device)
                        one_set_return = self.G(one_seg, l).detach().cpu().numpy()
                        one_set_return = np.squeeze(one_set_return)
                        one_set_return = norm.backward_process(one_set_return, target)
                        convert_result.append(one_set_return)

                    convert_con = np.concatenate(convert_result, axis=1)
                    convert_con = convert_con[:, 0 : content["coded_sp_norm"].shape[1]]
                    contigu = np.ascontiguousarray(convert_con.T, dtype=np.float64)
                    decoded_sp = decode_spectral_envelope(
                        contigu, SAMPLE_RATE, fft_size=FFTSIZE
                    )
                    f0_converted = norm.pitch_conversion(f0, speaker, target)
                    wav = synthesize(f0_converted, decoded_sp, ap, SAMPLE_RATE)

                    name = f"{speaker}-{target}_{filename}"
                    path = os.path.join(self.result_dir, name)
                    print(f"[save]:{path}")
                    sf.write(path, wav, SAMPLE_RATE)
