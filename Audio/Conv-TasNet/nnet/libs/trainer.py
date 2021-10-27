"""
@Yingzhao <yinger_z@126.com>
"""

import os
import shutil
import re
import time
import numpy as np
from itertools import permutations
import oneflow as flow
import oneflow.nn.functional as F
from oneflow.nn.utils import clip_grad_norm_
import matplotlib.pyplot as plt


def load_obj(obj, device):
    """
    Offload tensor object in obj to cuda device
    """

    def cuda(obj):
        return obj.to(device) if isinstance(obj, flow.Tensor) else obj

    if isinstance(obj, dict):
        return {key: load_obj(obj[key], device) for key in obj}
    elif isinstance(obj, list):
        return [load_obj(val, device) for val in obj]
    else:
        return cuda(obj)


class Trainer(object):
    def __init__(
        self,
        nnet,
        device,
        args,
        optimizer="adam",
        optimizer_kwargs=None,
        clip_norm=None,
        min_lr=0,
        patience=0,
        factor=0.5,
        no_impr=6,
    ):
        if not flow.cuda.is_available():
            raise RuntimeError("CUDA device unavailable...exist")
        self.model = nnet
        self.optimizer = self.create_optimizer(optimizer, optimizer_kwargs)

        # Training config
        self.device = device
        self.start_epoch = 0
        self.val_no_impv = 0
        self.epochs = args.epochs
        self.half_lr = args.half_lr
        self.early_stop = args.early_stop
        self.clip_norm = clip_norm
        # save and load model
        self.save_folder = args.save_folder
        self.model_path = args.model_path
        self.checkpoint = args.checkpoint
        # logging
        self.print_freq = args.print_freq

        # Create save folder
        os.makedirs(self.save_folder, exist_ok=True)
        self.prev_val_loss = float("inf")
        self.best_val_loss = float("inf")
        self.halving = False

        self.num_params = (
            sum([param.nelement() for param in nnet.parameters()]) / 10.0 ** 6
        )

    def create_optimizer(self, optimizer, kwargs, state=None):
        supported_optimizer = {
            "sgd": flow.optim.SGD,
            "rmsprop": flow.optim.RMSprop,
            "adam": flow.optim.Adam,
        }
        if optimizer not in supported_optimizer:
            raise ValueError("Now only support optimizer {}".format(optimizer))
        opt = supported_optimizer[optimizer](self.model.parameters(), **kwargs)

        return opt

    def compute_loss(self, est, egs):
        raise NotImplementedError

    def _run_one_epoch(self, epoch, tr_loader, cv_loader, cross_valid=False):
        start = time.time()
        total_loss = 0
        data_loader = tr_loader if not cross_valid else cv_loader

        for i, egs in enumerate(data_loader):
            # load to gpu
            egs = load_obj(egs, self.device)
            est = self.model(egs["mix"])
            loss = self.compute_loss(est, egs)

            if not cross_valid:
                self.optimizer.zero_grad()
                loss.backward()
                if self.clip_norm:
                    clip_grad_norm_(self.model.parameters(), self.clip_norm)
                self.optimizer.step()

            total_loss += float(loss.numpy())

            if i % self.print_freq == 0:
                print(
                    "Epoch {0} | Iter {1} | Average Loss {2:.3f} | "
                    "Current Loss {3:.6f} | {4:.1f} ms/batch".format(
                        epoch + 1,
                        i + 1,
                        total_loss / (i + 1),
                        float(loss.numpy()),
                        1000 * (time.time() - start) / (i + 1),
                    ),
                    flush=True,
                )

        return total_loss / (i + 1)

    def train(self, tr_loader, cv_loader):
        # Train model multi-epoches
        train_losses = []
        val_losses = []
        plt.title("Loss of train and test")
        for epoch in range(self.start_epoch, self.epochs):
            # Train one epoch
            print("Training...")
            self.model.train()
            start = time.time()
            tr_avg_loss = self._run_one_epoch(epoch, tr_loader, cv_loader)
            train_losses.append(tr_avg_loss)
            print("-" * 85)
            print(
                "Train Summary | End of Epoch {0} | Time {1:.2f}s | "
                "Train Loss {2:.3f}".format(epoch + 1, time.time() - start, tr_avg_loss)
            )
            print("-" * 85)

            # Save model each epoch
            if self.checkpoint:
                file_path = os.path.join(
                    self.save_folder, "epoch%d.pth.tar" % (epoch + 1)
                )
                flow.save(self.model.state_dict(), file_path)
                print("Saving checkpoint model to %s" % file_path)
                for dirs in os.listdir(self.save_folder):
                    dir_name = os.path.join(self.save_folder, dirs)
                    dir = dir_name.split("/")[-1]
                    dir = re.findall(r"\d+", dir)
                    if dir == []:
                        dir = 1000
                    else:
                        dir = int(dir[0])
                    if (epoch + 1) - dir >= 5:
                        shutil.rmtree(dir_name)

            # Cross validation
            print("Cross validation...")
            self.model.eval()
            val_loss = self._run_one_epoch(
                epoch, tr_loader, cv_loader, cross_valid=True
            )
            val_losses.append(val_loss)
            print("-" * 85)
            print(
                "Valid Summary | End of Epoch {0} | Time {1:.2f}s | "
                "Valid Loss {2:.3f}".format(epoch + 1, time.time() - start, val_loss)
            )
            print("-" * 85)
            # Adjust learning rate (halving)
            if self.half_lr:
                if val_loss >= self.prev_val_loss:
                    self.val_no_impv += 1
                    if self.val_no_impv >= 2:
                        self.halving = True
                    if self.val_no_impv >= 10 and self.early_stop:
                        print("No imporvement for 10 epochs, early stopping.")
                        break
                else:
                    self.val_no_impv = 0
            if self.halving:
                for param_group in self.optimizer.param_groups:
                    param_group["lr"] /= 2.0
                print(
                    "Learning rate adjusted to: {lr:.6f}".format(lr=param_group["lr"])
                )
                self.halving = False
            self.prev_val_loss = val_loss

            # Save the best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                file_path = os.path.join(self.save_folder, self.model_path)
                if os.path.exists(file_path):
                    shutil.rmtree(file_path)
                flow.save(self.model.state_dict(), file_path)
                np.save(file_path + "/epoch.npy", epoch)

                print("Find better validated model, saving to %s" % file_path)

        # loss image
        x = [i for i in range(self.epochs)]
        plt.plot(x, train_losses, "b-", label=u"train_loss", linewidth=0.8)
        plt.plot(x, val_losses, "c-", label=u"val_loss", linewidth=0.8)
        plt.legend()
        # plt.xticks(l, lx)
        plt.ylabel("loss")
        plt.xlabel("epoch")
        plt.savefig("conv_tasnet_loss.png")


class SiSnrTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super(SiSnrTrainer, self).__init__(*args, **kwargs)

    def sisnr(self, x, s, eps=1e-8):
        """
        Arguments:
        x: separated signal, N x S tensor
        s: reference signal, N x S tensor
        Return:
        sisnr: N tensor
        """

        def l2norm(mat, keepdim=False):
            return flow.linalg.norm(mat, dim=-1, keepdim=keepdim)

        if x.shape != s.shape:
            raise RuntimeError(
                "Dimention mismatch when calculate si-snr, {} vs {}".format(
                    x.shape, s.shape
                )
            )
        x_zm = x - flow.mean(x, dim=-1, keepdim=True)
        s_zm = s - flow.mean(s, dim=-1, keepdim=True)
        t = (
            flow.sum(x_zm * s_zm, dim=-1, keepdim=True)
            * s_zm
            / (l2norm(s_zm, keepdim=True) ** 2 + eps)
        )

        res = 20 * flow.log(eps + l2norm(t) / (l2norm(x_zm - t) + eps)) / 2.3025851

        return res

    def compute_loss(self, est, egs):
        # spks x n x S
        ests = est
        # spks x n x S
        refs = egs["ref"]
        num_spks = len(refs)

        def sisnr_loss(permute):
            # for one permute
            return sum(
                [self.sisnr(ests[s], refs[t]) for s, t in enumerate(permute)]
            ) / len(permute)

        # P x N
        N = egs["mix"].size(0)
        sisnr_mat = flow.stack([sisnr_loss(p) for p in permutations(range(num_spks))])
        max_perutt, _ = flow.max(sisnr_mat, dim=0)
        # si-snr
        return -flow.sum(max_perutt) / N
