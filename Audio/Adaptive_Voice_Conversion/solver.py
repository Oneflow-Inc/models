import oneflow as flow
import os
import oneflow.nn as nn
import yaml
from model import AE
from data_utils import get_data_loader
from data_utils import PickleDataset
from utils import *
import time
import re
import shutil


class Solver(object):
    def __init__(self, config, args):
        # config store the value of hyperparameters, turn to attr by AttrDict
        self.config = config
        print(config)

        # args store other information
        self.args = args
        print(self.args)

        # Create save folder
        os.makedirs(self.args.store_model_path, exist_ok=True)

        # get dataloader
        self.get_data_loaders()

        # init the model with config
        self.build_model()
        self.save_config()

    def save_config(self):
        with open(f"{self.args.store_model_path}.config.yaml", "w") as f:
            yaml.dump(self.config, f)
        with open(f"{self.args.store_model_path}.args.yaml", "w") as f:
            yaml.dump(vars(self.args), f)
        return

    def get_data_loaders(self):
        data_dir = self.args.data_dir
        self.train_dataset = PickleDataset(
            os.path.join(data_dir, f"{self.args.train_set}.pkl"),
            os.path.join(data_dir, self.args.train_index_file),
            segment_size=self.config["data_loader"]["segment_size"],
        )
        self.train_loader = get_data_loader(
            self.train_dataset,
            frame_size=self.config["data_loader"]["frame_size"],
            batch_size=self.config["data_loader"]["batch_size"],
            shuffle=self.config["data_loader"]["shuffle"],
            num_workers=0,
            drop_last=False,
        )
        self.train_iter = infinite_iter(self.train_loader)
        return

    def build_model(self):
        # create model, discriminator, optimizers
        self.model = cc(AE(self.config))
        print(self.model)
        optimizer = self.config["optimizer"]
        self.opt = flow.optim.Adam(
            self.model.parameters(),
            lr=optimizer["lr"],
            betas=(optimizer["beta1"], optimizer["beta2"]),
            amsgrad=optimizer["amsgrad"],
            weight_decay=optimizer["weight_decay"],
        )
        return

    def ae_step(self, data, lambda_kl):
        x = cc(data)
        mu, log_sigma, emb, dec = self.model(x)
        criterion = nn.L1Loss()
        loss_rec = criterion(dec, x)
        loss_kl = 0.5 * flow.mean(
            flow.exp(log_sigma) + flow.mul(mu, mu) - 1 - log_sigma
        )
        loss = self.config["lambda"]["lambda_rec"] * loss_rec + lambda_kl * loss_kl
        self.opt.zero_grad()
        loss.backward()
        grad_norm = flow.nn.utils.clip_grad_norm_(
            self.model.parameters(), max_norm=self.config["optimizer"]["grad_norm"]
        )
        self.opt.step()
        meta = {
            "loss_rec": loss_rec.item(),
            "loss_kl": loss_kl.item(),
            "loss": loss.item(),
            "grad_norm": grad_norm,
        }
        return meta

    def train(self, n_iterations):
        start = time.time()
        for iteration in range(n_iterations):
            if iteration >= self.config["annealing_iters"]:
                lambda_kl = self.config["lambda"]["lambda_kl"]
            else:
                lambda_kl = (
                    self.config["lambda"]["lambda_kl"]
                    * (iteration + 1)
                    / self.config["annealing_iters"]
                )

            data = next(self.train_iter)
            meta = self.ae_step(data, lambda_kl)

            if iteration % self.args.summary_steps == 0:
                print(
                    "Iter {0} | loss_kl {1:.3f} | "
                    "loss_rec {2:.3f} | loss {3:.3f}".format(
                        iteration, meta["loss_kl"], meta["loss_rec"], meta["loss"],
                    ),
                    flush=True,
                )

            if (
                iteration + 1
            ) % self.args.save_steps == 0 or iteration + 1 == n_iterations:
                file_path = os.path.join(
                    self.args.store_model_path, "iteration%d.pth.tar" % (iteration + 1)
                )
                flow.save(self.model.state_dict(), file_path)
                print("Saving checkpoint model to %s" % file_path)
                for dirs in os.listdir(self.args.store_model_path):
                    dir_name = os.path.join(self.args.store_model_path, dirs)
                    dir = dir_name.split("/")[-1]
                    dir = re.findall(r"\d+", dir)
                    if dir == []:
                        dir = 100000000
                    else:
                        dir = int(dir[0])
                    if (iteration + 1) - dir >= 24999:
                        shutil.rmtree(dir_name)
        print("Train Time {0:.2f}s".format(time.time() - start))
        return
