import oneflow as flow
import oneflow.nn as nn

from model.dnn_models import LayerNorm, SincConv_fast
from utils.data_utils import act_fun


class SincNet(nn.Module):
    def __init__(self, options):
        super(SincNet, self).__init__()

        self.cnn_N_filt = options["cnn_N_filt"]
        self.cnn_len_filt = options["cnn_len_filt"]
        self.cnn_max_pool_len = options["cnn_max_pool_len"]

        self.cnn_act = options["cnn_act"]
        self.cnn_drop = options["cnn_drop"]

        self.cnn_use_laynorm = options["cnn_use_laynorm"]
        self.cnn_use_batchnorm = options["cnn_use_batchnorm"]
        self.cnn_use_laynorm_inp = options["cnn_use_laynorm_inp"]
        self.cnn_use_batchnorm_inp = options["cnn_use_batchnorm_inp"]

        self.input_dim = int(options["input_dim"])

        self.fs = options["fs"]

        self.N_cnn_lay = len(options["cnn_N_filt"])
        self.conv = nn.ModuleList([])
        self.bn = nn.ModuleList([])
        self.ln = nn.ModuleList([])
        self.act = nn.ModuleList([])
        self.drop = nn.ModuleList([])

        if self.cnn_use_laynorm_inp:
            self.ln0 = LayerNorm(self.input_dim)

        if self.cnn_use_batchnorm_inp:
            self.bn0 = nn.BatchNorm1d([self.input_dim], momentum=0.05)

        current_input = self.input_dim

        for i in range(self.N_cnn_lay):

            N_filt = int(self.cnn_N_filt[i])
            len_filt = int(self.cnn_len_filt[i])

            # dropout
            self.drop.append(nn.Dropout(p=self.cnn_drop[i]))

            # activation
            self.act.append(act_fun(self.cnn_act[i]))

            # layer norm initialization
            self.ln.append(
                LayerNorm(
                    (
                        N_filt,
                        int(
                            (current_input - self.cnn_len_filt[i] + 1)
                            / self.cnn_max_pool_len[i]
                        ),
                    )
                )
            )

            self.bn.append(
                nn.BatchNorm1d(
                    N_filt,
                    int(
                        (current_input - self.cnn_len_filt[i] + 1)
                        / self.cnn_max_pool_len[i]
                    ),
                    momentum=0.05,
                )
            )

            if i == 0:
                self.conv.append(
                    SincConv_fast(self.cnn_N_filt[0], self.cnn_len_filt[0], self.fs)
                )
            else:
                self.conv.append(
                    nn.Conv1d(
                        self.cnn_N_filt[i - 1], self.cnn_N_filt[i], self.cnn_len_filt[i]
                    )
                )

            current_input = int(
                (current_input - self.cnn_len_filt[i] + 1) / self.cnn_max_pool_len[i]
            )

        self.out_dim = current_input * N_filt

    def forward(self, x):
        batch = x.shape[0]
        seq_len = x.shape[1]

        if bool(self.cnn_use_laynorm_inp):
            x = self.ln0((x))

        if bool(self.cnn_use_batchnorm_inp):
            x = self.bn0((x))

        x = x.reshape(batch, 1, seq_len)

        for i in range(self.N_cnn_lay):

            if self.cnn_use_laynorm[i]:
                if i == 0:
                    max_pool1 = nn.MaxPool1d(
                        self.cnn_max_pool_len[i], stride=None, padding=0, dilation=1
                    )
                    x = max_pool1(flow.abs(self.conv[i](x)))
                    x = self.drop[i](self.act[i](self.ln[i](x)))
                else:
                    max_pool2 = nn.MaxPool1d(
                        self.cnn_max_pool_len[i], stride=None, padding=0, dilation=1
                    )
                    x = max_pool2(self.conv[i](x))
                    x = self.drop[i](self.act[i](self.ln[i](x)))

            if self.cnn_use_batchnorm[i]:
                max_pool3 = nn.MaxPool1d(
                    self.cnn_max_pool_len[i], stride=None, padding=0, dilation=1
                )
                x = max_pool3(self.conv[i](x))
                x = self.drop[i](self.act[i](self.bn[i](x)))

            if self.cnn_use_batchnorm[i] == False and self.cnn_use_laynorm[i] == False:
                max_pool4 = nn.MaxPool1d(
                    self.cnn_max_pool_len[i], stride=None, padding=0, dilation=1
                )
                x = max_pool4(self.conv[i](x))
                x = self.drop[i](self.act[i](x))

        x = x.reshape(batch, -1)

        return x
