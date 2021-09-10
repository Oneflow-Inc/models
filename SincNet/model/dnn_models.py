import sys
import math

import numpy as np
import oneflow as flow
import oneflow.nn as nn
import oneflow.nn.functional as F

from utils.data_utils import flip, sinc, act_fun


class SincConv_fast(nn.Module):
    """Sinc-based convolution
    Parameters
    ----------
    in_channels : `int`
        Number of input channels. Must be 1.
    out_channels : `int`
        Number of filters.
    kernel_size : `int`
        Filter length.
    sample_rate : `int`, optional
        Sample rate. Defaults to 16000.
    Usage
    -----
    See `torch.nn.Conv1d`
    Reference
    ---------
    Mirco Ravanelli, Yoshua Bengio,
    "Speaker Recognition from raw waveform with SincNet".
    https://arxiv.org/abs/1808.00158
    """

    @staticmethod
    def to_mel(hz):
        return 2595 * np.log10(1 + hz / 700)

    @staticmethod
    def to_hz(mel):
        return 700 * (10 ** (mel / 2595) - 1)

    def __init__(
        self,
        out_channels,
        kernel_size,
        sample_rate=16000,
        in_channels=1,
        stride=1,
        padding=0,
        dilation=1,
        bias=False,
        groups=1,
        min_low_hz=50,
        min_band_hz=50,
    ):

        super(SincConv_fast, self).__init__()

        if in_channels != 1:
            msg = (
                "SincConv only support one input channel (here, in_channels = {%i})"
                % (in_channels)
            )
            raise ValueError(msg)

        self.out_channels = out_channels
        self.kernel_size = kernel_size

        # Forcing the filters to be odd (i.e, perfectly symmetrics)
        if kernel_size % 2 == 0:
            self.kernel_size = self.kernel_size + 1

        self.stride = stride
        self.padding = padding
        self.dilation = dilation

        if bias:
            raise ValueError("SincConv does not support bias.")
        if groups > 1:
            raise ValueError("SincConv does not support groups.")

        self.sample_rate = sample_rate
        self.min_low_hz = min_low_hz
        self.min_band_hz = min_band_hz

        # initialize filterbanks such that they are equally spaced in Mel scale
        low_hz = 30
        high_hz = self.sample_rate / 2 - (self.min_low_hz + self.min_band_hz)

        mel = np.linspace(
            self.to_mel(low_hz), self.to_mel(high_hz), self.out_channels + 1
        )
        hz = self.to_hz(mel)

        # filter lower frequency (out_channels, 1)
        self.low_hz_ = nn.Parameter(flow.Tensor(hz[:-1]).reshape(-1, 1))

        # filter frequency band (out_channels, 1)
        self.band_hz_ = nn.Parameter(flow.Tensor(np.diff(hz)).reshape(-1, 1))

        # Hamming window
        n_lin = flow.Tensor(
            np.linspace(0, (self.kernel_size / 2) - 1, int((self.kernel_size / 2)))
        )
        self.window_ = 0.54 - 0.46 * flow.cos(2 * math.pi * n_lin / self.kernel_size)

        # (1, kernel_size/2)
        n = (self.kernel_size - 1) / 2.0
        self.n_ = (
            2
            * math.pi
            * flow.Tensor(np.arange(-n, 0).reshape(1, -1) / self.sample_rate)
        )

    def forward(self, waveforms):
        """
        Parameters
        ----------
        waveforms : `oneflow.Tensor` (batch_size, 1, n_samples)
            Batch of waveforms.
        Returns
        -------
        features : `oneflow.Tensor` (batch_size, out_channels, n_samples_out)
            Batch of sinc filters activations.
        """
        self.n_ = self.n_.to(waveforms.device)

        self.window_ = self.window_.to(waveforms.device)

        low = self.min_low_hz + flow.abs(self.low_hz_)

        high = flow.clamp(
            low + self.min_band_hz + flow.abs(self.band_hz_),
            self.min_low_hz,
            self.sample_rate / 2,
        )
        band = (high - low)[:, 0]

        f_times_t_low = flow.matmul(low, self.n_)
        f_times_t_high = flow.matmul(high, self.n_)

        band_pass_left = (
            (flow.sin(f_times_t_high) - flow.sin(f_times_t_low)) / (self.n_ / 2)
        ) * self.window_
        band_pass_center = 2 * band.reshape(-1, 1)
        band_pass_right = flow.flip(band_pass_left, dims=[1])

        band_pass = flow.cat([band_pass_left, band_pass_center, band_pass_right], dim=1)

        band_pass = band_pass / (2 * band[:, None])

        self.filters = (band_pass).reshape(self.out_channels, 1, self.kernel_size)

        output = F.conv1d(
            waveforms,
            self.filters,
            stride=[self.stride],
            padding=[self.padding],
            dilation=[self.dilation],
            bias=None,
            groups=1,
        )
        return output


class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(flow.ones(features))
        self.beta = nn.Parameter(flow.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta


class MLP(nn.Module):
    def __init__(self, options):
        super(MLP, self).__init__()

        self.input_dim = int(options["input_dim"])
        self.fc_lay = options["fc_lay"]
        self.fc_drop = options["fc_drop"]
        self.fc_use_batchnorm = options["fc_use_batchnorm"]
        self.fc_use_laynorm = options["fc_use_laynorm"]
        self.fc_use_laynorm_inp = options["fc_use_laynorm_inp"]
        self.fc_use_batchnorm_inp = options["fc_use_batchnorm_inp"]
        self.fc_act = options["fc_act"]

        self.wx = nn.ModuleList([])
        self.bn = nn.ModuleList([])
        self.ln = nn.ModuleList([])
        self.act = nn.ModuleList([])
        self.drop = nn.ModuleList([])

        # input layer normalization
        if self.fc_use_laynorm_inp:
            self.ln0 = LayerNorm(self.input_dim)

        # input batch normalization
        if self.fc_use_batchnorm_inp:
            self.bn0 = nn.BatchNorm1d([self.input_dim], momentum=0.05)

        self.N_fc_lay = len(self.fc_lay)

        current_input = self.input_dim

        # Initialization of hidden layers
        for i in range(self.N_fc_lay):

            # dropout
            self.drop.append(nn.Dropout(p=self.fc_drop[i]))

            # activation
            self.act.append(act_fun(self.fc_act[i]))

            add_bias = True

            # layer norm initialization
            self.ln.append(LayerNorm(self.fc_lay[i]))
            self.bn.append(nn.BatchNorm1d(self.fc_lay[i], momentum=0.05))

            if self.fc_use_laynorm[i] or self.fc_use_batchnorm[i]:
                add_bias = False

            # Linear operations
            self.wx.append(nn.Linear(current_input, self.fc_lay[i], bias=add_bias))

            # weight initialization
            self.wx[i].weight = nn.Parameter(
                flow.Tensor(self.fc_lay[i], current_input).uniform_(
                    -np.sqrt(0.01 / (current_input + self.fc_lay[i])),
                    np.sqrt(0.01 / (current_input + self.fc_lay[i])),
                )
            )
            self.wx[i].bias = nn.Parameter(flow.zeros(self.fc_lay[i]))

            current_input = self.fc_lay[i]

    def forward(self, x):

        # Applying Layer/Batch Norm
        if bool(self.fc_use_laynorm_inp):
            x = self.ln0((x))

        if bool(self.fc_use_batchnorm_inp):
            x = self.bn0((x))

        for i in range(self.N_fc_lay):
            if self.fc_act[i] != "linear":
                if self.fc_use_laynorm[i]:
                    x = self.drop[i](self.act[i](self.ln[i](self.wx[i](x))))

                if self.fc_use_batchnorm[i]:
                    x = self.drop[i](self.act[i](self.bn[i](self.wx[i](x))))

                if (
                    self.fc_use_batchnorm[i] == False
                    and self.fc_use_laynorm[i] == False
                ):
                    x = self.drop[i](self.act[i](self.wx[i](x)))
            else:
                if self.fc_use_laynorm[i]:
                    x = self.drop[i](self.ln[i](self.wx[i](x)))

                if self.fc_use_batchnorm[i]:
                    x = self.drop[i](self.bn[i](self.wx[i](x)))

                if (
                    self.fc_use_batchnorm[i] == False
                    and self.fc_use_laynorm[i] == False
                ):
                    x = self.drop[i](self.wx[i](x))
        return x
