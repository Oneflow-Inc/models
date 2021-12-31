import re
import os
import yaml
import random
import logging
import shutil
import numpy as np
import oneflow as flow
import argparse
from otrans.model import End2EndModel, LanguageModel
from otrans.train.scheduler import BuildOptimizer, BuildScheduler
from otrans.train.trainer import Trainer
from otrans.utils import count_parameters
from otrans.data.loader import FeatureLoader


def main(args, params, expdir):

    model_type = params["model"]["type"]
    if model_type[-2:] == "lm":
        model = LanguageModel[model_type](params["model"])
    else:
        model = End2EndModel[model_type](params["model"])

    # Count total parameters
    count_parameters(model.named_parameters())

    if args.ngpu >= 1:
        model.cuda()
    logging.info(model)

    optimizer = BuildOptimizer[params["train"]["optimizer_type"]](
        filter(lambda p: p.requires_grad, model.parameters()),
        **params["train"]["optimizer"]
    )
    logger.info("[Optimizer] Build a %s optimizer!" % params["train"]["optimizer_type"])
    scheduler = BuildScheduler[params["train"]["scheduler_type"]](
        optimizer, **params["train"]["scheduler"]
    )
    logger.info("[Scheduler] Build a %s scheduler!" % params["train"]["scheduler_type"])

    trainer = Trainer(
        params,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        expdir=expdir,
        ngpu=args.ngpu,
        local_rank=args.local_rank,
        is_debug=args.debug,
        keep_last_n_chkpt=args.keep_last_n_chkpt,
        from_epoch=args.from_epoch,
    )

    train_loader = FeatureLoader(params, "train", ngpu=args.ngpu)

    trainer.train(train_loader=train_loader)


import multiprocessing as mp

mp.set_start_method("spawn", True)
parser = argparse.ArgumentParser()
parser.add_argument(
    "-c", "--config", type=str, default="egs/aishell/conf/transformer_baseline.yaml"
)
parser.add_argument("-n", "--ngpu", type=int, default=1)
parser.add_argument("-g", "--gpus", type=str, default="0")
parser.add_argument("-r", "--local_rank", type=int, default=0)
parser.add_argument(
    "-l", "--logging_level", type=str, default="info", choices=["info", "debug"]
)
parser.add_argument("-lg", "--log_file", type=str, default=None)
parser.add_argument("-dir", "--expdir", type=str, default=None)
parser.add_argument("-debug", "--debug", action="store_true", default=False)
parser.add_argument("-knpt", "--keep_last_n_chkpt", type=int, default=30)
parser.add_argument("-tfs", "--from_step", type=int, default=0)
parser.add_argument("-tfe", "--from_epoch", type=int, default=0)

cmd_args = parser.parse_args()

with open(cmd_args.config, "r") as f:
    params = yaml.load(f, Loader=yaml.FullLoader)

if cmd_args.expdir is not None:
    expdir = os.path.join(cmd_args.expdir, params["train"]["save_name"])
else:
    expdir = os.path.join(
        "egs", params["data"]["name"], "exp", params["train"]["save_name"]
    )
if not os.path.exists(expdir):
    os.makedirs(expdir)

shutil.copy(cmd_args.config, os.path.join(expdir, "config.yaml"))

logging_level = {"info": logging.INFO, "debug": logging.DEBUG}

if cmd_args.log_file is not None:
    log_file = cmd_args.log_file
else:
    log_file = cmd_args.config.split("/")[-1][:-5] + ".log"

LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
logging.basicConfig(level=logging_level[cmd_args.logging_level], format=LOG_FORMAT)
logger = logging.getLogger(__name__)

if cmd_args.ngpu == 1:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(cmd_args.gpus)
    logger.info("Set CUDA_VISIBLE_DEVICES as %s" % cmd_args.gpus)

main(cmd_args, params, expdir)
