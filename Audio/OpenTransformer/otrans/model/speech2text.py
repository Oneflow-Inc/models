from logging import log
import os
import math
import oneflow as flow
import logging
import oneflow.nn as nn
from otrans.model.base import BaseModel
from otrans.frontend import BuildFrontEnd
from otrans.encoder import BuildEncoder
from otrans.decoder import BuildDecoder
from otrans.module.loss import LabelSmoothingLoss
from otrans.model.ctc import CTCAssistor

logger = logging.getLogger(__name__)


class SpeechToText(BaseModel):
    def __init__(self, params):
        super(SpeechToText, self).__init__()

        self.frontend = BuildFrontEnd[params["frontend_type"]](**params["frontend"])
        logger.info("Build a %s frontend!" % params["frontend_type"])
        self.encoder = BuildEncoder[params["encoder_type"]](**params["encoder"])
        logger.info("Build a %s encoder!" % params["encoder_type"])
        self.decoder = BuildDecoder[params["decoder_type"]](**params["decoder"])
        logger.info("Build a %s decoder!" % params["decoder_type"])

        self.crit = LabelSmoothingLoss(
            size=params["decoder"]["vocab_size"], smoothing=params["smoothing"]
        )

        self.ctc_weight = params["ctc_weight"]
        if self.ctc_weight > 0.0:
            self.assistor = CTCAssistor(
                hidden_size=params["encoder_output_size"],
                vocab_size=params["decoder"]["vocab_size"],
                lookahead_steps=params["lookahead_steps"]
                if "lookahead_steps" in params
                else 0,
            )
            logger.info("Build a CTC Assistor with weight %.2f" % self.ctc_weight)

    def forward(self, inputs, targets):

        enc_inputs = inputs["inputs"]
        enc_mask = inputs["mask"]

        truth = targets["targets"]
        truth_length = targets["targets_length"]

        enc_inputs, enc_mask = self.frontend(enc_inputs, enc_mask)

        # 1. forward encoder
        memory, memory_mask, _ = self.encoder(enc_inputs, enc_mask)

        # 2. forward decoder
        target_in = truth[:, :-1].clone()
        logits, _ = self.decoder(target_in, memory, memory_mask)

        # 3. compute attention loss
        target_out = truth[:, 1:].clone()
        loss = self.crit(logits, target_out)

        if self.ctc_weight > 0:
            loss_ctc = self.compute_ctc_loss(
                memory, memory_mask, target_out, truth_length
            )
            return (
                (1 - self.ctc_weight) * loss + self.ctc_weight * loss_ctc,
                {"CTCLoss": loss_ctc.item()},
            )
        else:
            return loss, None

    def compute_ctc_loss(self, memory, memory_mask, targets_out, targets_length):
        memory_length = flow.sum(memory_mask.squeeze(1), dim=-1)
        loss_ctc = self.assistor(memory, memory_length, targets_out, targets_length)
        return loss_ctc

    def save_checkpoint(self, params, name):
        flow.save(params, os.path.join(name, "params.tar"))
        flow.save(self.frontend.state_dict(), os.path.join(name, "frontend.pt"))
        flow.save(self.encoder.state_dict(), os.path.join(name, "encoder.pt"))
        flow.save(self.decoder.state_dict(), os.path.join(name, "decoder.pt"))
        if self.ctc_weight > 0.0:
            flow.save(self.assistor.state_dict(), os.path.join(name, "ctc.pt"))

    def load_model(self, chkpt1, chkpt2, chkpt3):
        self.frontend.load_state_dict(flow.load(chkpt1))
        self.encoder.load_state_dict(flow.load(chkpt2))
        self.decoder.load_state_dict(flow.load(chkpt3))

    def set_epoch(self, epoch):
        pass
