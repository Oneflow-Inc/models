import json
import argparse

import oneflow as flow
import numpy as np
import soundfile as sf

from model.model import simple_CNN


def get_args():
    parser = argparse.ArgumentParser("""Speaker Identification Demo Infer""")
    parser.add_argument(
        "--label_dict", type=str, default="data_preprocessed/label_dict.json"
    )
    parser.add_argument("--num_speakers", type=int, default=2)
    parser.add_argument("--load_path", type=str, default="save_models/CNN_model")

    args = parser.parse_args()
    return args


def example_precess(wav, lab, wlen=3200, fact_amp=0.2):
    np.random.seed(10)

    sig_batch = np.zeros([1, wlen])
    lab_batch = np.zeros(1)
    rand_amp_arr = np.random.uniform(1.0 - fact_amp, 1 + fact_amp, 1)

    [signal, fs] = sf.read(wav)

    snt_len = signal.shape[0]
    snt_beg = np.random.randint(snt_len - wlen - 1)
    snt_end = snt_beg + wlen

    channels = len(signal.shape)
    if channels == 2:
        print("WARNING: stereo to mono")
        signal = signal[:, 0]

    sig_batch[0, :] = signal[snt_beg:snt_end] * rand_amp_arr[0]
    lab_batch[0] = int(lab)

    inp = flow.Tensor(sig_batch, dtype=flow.float32).to("cuda")
    lab = flow.Tensor(lab_batch, dtype=flow.float32).to("cuda")

    return inp, lab


def infer(opt):
    with open(opt.label_dict, "r") as f:
        lab_dict = json.load(f)

    cnn = simple_CNN(opt.num_speakers)
    cnn.to("cuda")

    cnn.load_state_dict(flow.load(opt.load_path))
    cnn.eval()

    label_list = lab_dict["test"]
    err_sum = 0
    for wav, label in label_list:
        inp, lab = example_precess(wav, label)
        inp = inp.unsqueeze(1)
        pout = cnn(inp)
        pred = flow.argmax(pout, dim=1)

        err = 1 if (pred + 1).numpy() != lab.long().numpy() else 0
        err_sum += err
        print(
            "wav_filename: ",
            wav,
            "    predicted speaker id: ",
            (pred + 1).numpy()[0],
            "    real speaker id: ",
            lab.long().numpy()[0],
        )
    print("accuracy: ", 1 - err_sum / 6)


if __name__ == "__main__":
    opt = get_args()
    infer(opt)
