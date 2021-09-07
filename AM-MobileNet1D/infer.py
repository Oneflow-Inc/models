"""
@author: Wang Yizhang <1739601638@qq.com>
"""
import os

import oneflow as flow
import oneflow.nn as nn
import soundfile as sf
import numpy as np

from utils.data_utils import ReadList, read_conf, str_to_bool
from model.mobilenet1d import MobileNetV2, AdditiveMarginSoftmax


# Reading cfg file
options = read_conf()

# [data]
te_lst = options.te_lst
class_dict_file = options.lab_dict
data_folder = options.data_folder + "/"
pretrain_models = options.pretrain_models

# [windowing]
fs = int(options.fs)
cw_len = int(options.cw_len)
cw_shift = int(options.cw_shift)

# [class]
class_lay = list(map(int, options.class_lay.split(",")))
class_drop = list(map(float, options.class_drop.split(",")))
class_use_laynorm_inp = str_to_bool(options.class_use_laynorm_inp)
class_use_batchnorm_inp = str_to_bool(options.class_use_batchnorm_inp)
class_use_batchnorm = list(map(str_to_bool, options.class_use_batchnorm.split(",")))
class_use_laynorm = list(map(str_to_bool, options.class_use_laynorm.split(",")))
class_act = list(map(str, options.class_act.split(",")))

# [optimization]
lr = float(options.lr)
batch_size = int(options.batch_size)
begin_epochs = int(options.begin_epochs)
N_epochs = int(options.N_epochs)
N_batches = int(options.N_batches)
N_eval_epoch = int(options.N_eval_epoch)
seed = int(options.seed)

# test list
wav_lst_te = ReadList(te_lst)
snt_te = len(wav_lst_te)

# setting seed
np.random.seed(seed)

# loss function
if options.AMSoftmax == "True":
    print("Using AMSoftmax loss function...")
    cost = AdditiveMarginSoftmax(margin=float(options.AMSoftmax_m))
else:
    print("Using Softmax loss function...")
    cost = nn.NLLLoss()

# Converting context and shift in samples
wlen = int(fs * cw_len / 1000.00)
wshift = int(fs * cw_shift / 1000.00)

# Batch_dev
Batch_dev = batch_size

# Loading label dictionary
lab_dict = np.load(class_dict_file, allow_pickle=True).item()

MOBILENET_net = MobileNetV2(num_classes=class_lay)
MOBILENET_net.to("cuda")

# Loading model
MOBILENET_net.load_state_dict(flow.load(os.path.join(pretrain_models, "MOBILENET_net")))


# Infering on test datasets
def infer():
    MOBILENET_net.eval()
    loss_sum = 0
    err_sum = 0
    err_sum_snt = 0

    with flow.no_grad():
        for i in range(snt_te):
            [signal, fs] = sf.read(data_folder + wav_lst_te[i])
            signal = signal.reshape(1, -1)
            lab_batch = lab_dict[wav_lst_te[i]]

            # split signals into chunks
            beg_samp = 0
            end_samp = wlen

            N_fr = int((signal.shape[1] - wlen) / (wshift))

            sig_arr = np.zeros([Batch_dev, 1, wlen])

            lab = (flow.zeros(N_fr + 1) + lab_batch).to("cuda").long()
            pout = flow.zeros(N_fr + 1, class_lay[-1], dtype=flow.float32).to("cuda")
            count_fr = 0
            count_fr_tot = 0
            while end_samp < signal.shape[1]:
                sig_arr[count_fr, :] = signal[:, beg_samp:end_samp]
                beg_samp = beg_samp + wshift
                end_samp = beg_samp + wlen
                count_fr = count_fr + 1
                count_fr_tot = count_fr_tot + 1
                if count_fr == Batch_dev:
                    inp = flow.Tensor(sig_arr).to("cuda")
                    pout[count_fr_tot - Batch_dev : count_fr_tot, :] = MOBILENET_net(
                        inp
                    )

                    count_fr = 0
                    sig_arr = np.zeros([Batch_dev, 1, wlen])

            if count_fr > 0:
                inp = flow.Tensor(sig_arr[0:count_fr]).to("cuda")

                pout[count_fr_tot - count_fr : count_fr_tot, :] = MOBILENET_net(inp)

            pred = flow.argmax(pout, dim=1)
            loss = cost(pout, lab.long())

            err = np.mean(pred.numpy() != lab.long().numpy())

            best_class = flow.argmax(flow.sum(pout, dim=0), dim=0)
            err_sum_snt = err_sum_snt + (best_class.numpy() != lab[0].numpy())

            loss_sum = loss_sum + loss.detach()
            err_sum = err_sum + err

        err_tot_dev_snt = err_sum_snt / snt_te
        loss_tot_dev = loss_sum / snt_te
        err_tot_dev = err_sum / snt_te

    print(
        "loss_te=%f err_te=%f err_te_snt=%f"
        % (loss_tot_dev.numpy(), err_tot_dev, err_tot_dev_snt)
    )


if __name__ == "__main__":
    infer()
