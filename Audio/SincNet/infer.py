"""
@author: Wang Yizhang <1739601638@qq.com>
"""
import os

import numpy as np
import soundfile as sf
import oneflow as flow
import oneflow.nn as nn
import oneflow.optim as optim

from model.dnn_models import MLP
from model.SincNet import SincNet as CNN
from utils.data_utils import ReadList, read_conf, str_to_bool, create_batches_rnd


# Reading cfg file
options = read_conf()

# [data]
tr_lst = options.tr_lst
te_lst = options.te_lst
class_dict_file = options.lab_dict
data_folder = options.data_folder + "/"
pretrain_models = options.pretrain_models

# [windowing]
fs = int(options.fs)
cw_len = int(options.cw_len)
cw_shift = int(options.cw_shift)

# [cnn]
cnn_N_filt = list(map(int, options.cnn_N_filt.split(",")))
cnn_len_filt = list(map(int, options.cnn_len_filt.split(",")))
cnn_max_pool_len = list(map(int, options.cnn_max_pool_len.split(",")))
cnn_use_laynorm_inp = str_to_bool(options.cnn_use_laynorm_inp)
cnn_use_batchnorm_inp = str_to_bool(options.cnn_use_batchnorm_inp)
cnn_use_laynorm = list(map(str_to_bool, options.cnn_use_laynorm.split(",")))
cnn_use_batchnorm = list(map(str_to_bool, options.cnn_use_batchnorm.split(",")))
cnn_act = list(map(str, options.cnn_act.split(",")))
cnn_drop = list(map(float, options.cnn_drop.split(",")))

# [dnn]
fc_lay = list(map(int, options.fc_lay.split(",")))
fc_drop = list(map(float, options.fc_drop.split(",")))
fc_use_laynorm_inp = str_to_bool(options.fc_use_laynorm_inp)
fc_use_batchnorm_inp = str_to_bool(options.fc_use_batchnorm_inp)
fc_use_batchnorm = list(map(str_to_bool, options.fc_use_batchnorm.split(",")))
fc_use_laynorm = list(map(str_to_bool, options.fc_use_laynorm.split(",")))
fc_act = list(map(str, options.fc_act.split(",")))

# [class]
class_lay = list(map(int, options.class_lay.split(",")))
class_drop = list(map(float, options.class_drop.split(",")))
class_use_laynorm_inp = str_to_bool(options.class_use_laynorm_inp)
class_use_batchnorm_inp = str_to_bool(options.class_use_batchnorm_inp)
class_use_batchnorm = list(map(str_to_bool, options.class_use_batchnorm.split(",")))
class_use_laynorm = list(map(str_to_bool, options.class_use_laynorm.split(",")))
class_act = list(map(str, options.class_act.split(",")))

# test list
wav_lst_te = ReadList(te_lst)
snt_te = len(wav_lst_te)

# loss function
cost = nn.NLLLoss()
cost.to("cuda")

# Converting context and shift in samples
wlen = int(fs * cw_len / 1000.00)  # 3200
wshift = int(fs * cw_shift / 1000.00)

# Batch_dev
Batch_dev = 128

# Feature extractor CNN
CNN_arch = {
    "input_dim": wlen,
    "fs": fs,
    "cnn_N_filt": cnn_N_filt,
    "cnn_len_filt": cnn_len_filt,
    "cnn_max_pool_len": cnn_max_pool_len,
    "cnn_use_laynorm_inp": cnn_use_laynorm_inp,
    "cnn_use_batchnorm_inp": cnn_use_batchnorm_inp,
    "cnn_use_laynorm": cnn_use_laynorm,
    "cnn_use_batchnorm": cnn_use_batchnorm,
    "cnn_act": cnn_act,
    "cnn_drop": cnn_drop,
}

CNN_net = CNN(CNN_arch)
CNN_net.to("cuda")

# Loading label dictionary
lab_dict = np.load(class_dict_file, allow_pickle=True).item()

DNN1_arch = {
    "input_dim": CNN_net.out_dim,
    "fc_lay": fc_lay,
    "fc_drop": fc_drop,
    "fc_use_batchnorm": fc_use_batchnorm,
    "fc_use_laynorm": fc_use_laynorm,
    "fc_use_laynorm_inp": fc_use_laynorm_inp,
    "fc_use_batchnorm_inp": fc_use_batchnorm_inp,
    "fc_act": fc_act,
}

DNN1_net = MLP(DNN1_arch)
DNN1_net.to("cuda")

DNN2_arch = {
    "input_dim": fc_lay[-1],
    "fc_lay": class_lay,
    "fc_drop": class_drop,
    "fc_use_batchnorm": class_use_batchnorm,
    "fc_use_laynorm": class_use_laynorm,
    "fc_use_laynorm_inp": class_use_laynorm_inp,
    "fc_use_batchnorm_inp": class_use_batchnorm_inp,
    "fc_act": class_act,
}

DNN2_net = MLP(DNN2_arch)
DNN2_net.to("cuda")

# Loading model
CNN_net.load_state_dict(flow.load(os.path.join(pretrain_models, "CNN")))
DNN1_net.load_state_dict(flow.load(os.path.join(pretrain_models, "DNN1")))
DNN2_net.load_state_dict(flow.load(os.path.join(pretrain_models, "DNN2")))


# Infering on test datasets
def infer():
    CNN_net.eval()
    DNN1_net.eval()
    DNN2_net.eval()
    loss_sum = 0
    err_sum = 0
    err_sum_snt = 0

    with flow.no_grad():
        for i in range(snt_te):
            [signal, fs] = sf.read(data_folder + wav_lst_te[i])

            signal = flow.Tensor(signal).to("cuda")
            lab_batch = lab_dict[wav_lst_te[i].lower()]

            # split signals into chunks
            beg_samp = 0
            end_samp = wlen

            N_fr = int((signal.shape[0] - wlen) / (wshift))

            sig_arr = flow.zeros((Batch_dev, wlen), dtype=flow.float32).to("cuda")
            lab = (flow.zeros(N_fr + 1) + lab_batch).to("cuda").long()
            pout = flow.zeros((N_fr + 1, class_lay[-1]), dtype=flow.float32).to("cuda")
            count_fr = 0
            count_fr_tot = 0
            while end_samp < signal.shape[0]:
                sig_arr[count_fr, :] = signal[beg_samp:end_samp]
                beg_samp = beg_samp + wshift
                end_samp = beg_samp + wlen
                count_fr = count_fr + 1
                count_fr_tot = count_fr_tot + 1
                if count_fr == Batch_dev:
                    inp = flow.Tensor(sig_arr).to(sig_arr.device)
                    pout[count_fr_tot - Batch_dev : count_fr_tot, :] = DNN2_net(
                        DNN1_net(CNN_net(inp))
                    )
                    count_fr = 0
                    sig_arr = flow.zeros((Batch_dev, wlen), dtype=flow.float32).to(
                        "cuda"
                    )

            if count_fr > 0:
                inp = flow.Tensor(sig_arr[0:count_fr]).to(sig_arr.device)
                pout[count_fr_tot - count_fr : count_fr_tot, :] = DNN2_net(
                    DNN1_net(CNN_net(inp))
                )

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
