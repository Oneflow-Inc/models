import oneflow as flow
import oneflow.nn as nn
import oneflow.nn.functional as F
from otrans.data import PAD


class LabelSmoothingLoss(nn.Module):
    def __init__(self, size, smoothing=0.1, padding_idx=PAD, normalize_length=True):
        super().__init__()

        self.padding_idx = padding_idx
        self.smoothing = smoothing
        self.size = size
        self.normalize_length = normalize_length

    def forward(self, logits, target, mask=None):
        """LabelSmoothing Function with Mask

        Args:
            logits ([tensor]): logits with shape [batch, length, vocab_size]
            target ([tensor]): target with shape [batch, length]
            mask ([tensor], optional): mask tensor (bool) with shape [batch, length]
        """
        assert logits.dim() == 3 and logits.size(-1) == self.size

        pad_mask = target == self.padding_idx
        if mask is not None:
            mask = (pad_mask.int() + mask.int()) > 0
        else:
            mask = pad_mask

        logits = logits.reshape(-1, self.size)
        with flow.no_grad():
            confidence = logits.clone()
            confidence.fill_(self.smoothing / (self.size - 1))
            confidence = flow.scatter(
                confidence, 1, target.reshape(-1).unsqueeze(1), 1 - self.smoothing
            )

        logsoftmax = nn.LogSoftmax(dim=-1)
        KLdiv = nn.KLDivLoss(reduction="none", log_target=False)
        loss = flow.sum(KLdiv(logsoftmax(logits), confidence), dim=-1)

        total = flow.sum(mask == 0)
        denom = total if self.normalize_length else logits.size(0)
        loss = flow.masked_fill(loss, mask.reshape(-1), 0.0)
        loss = flow.sum(loss) / denom

        return loss
