import json

import numpy as np
import soundfile as sf
import oneflow as flow


def create_batches_rnd(lab_dict, batch_size=32, wlen=3200, fact_amp=0.2, train=True):

    label_list = lab_dict["train"] if train else lab_dict["test"]
    N_snt = len(label_list)

    # Initialization of the minibatch (batch_size,[0=>x_t,1=>x_t+N,1=>random_samp])
    sig_batch = np.zeros([batch_size, wlen])
    lab_batch = np.zeros(batch_size)

    snt_id_arr = np.random.randint(N_snt, size=batch_size)
    rand_amp_arr = np.random.uniform(1.0 - fact_amp, 1 + fact_amp, batch_size)

    for i in range(batch_size):

        [signal, fs] = sf.read(label_list[snt_id_arr[i]][0])

        # accesing to a random chunk
        snt_len = signal.shape[0]
        snt_beg = np.random.randint(snt_len - wlen - 1)  # randint(0, snt_len-2*wlen-1)
        snt_end = snt_beg + wlen

        channels = len(signal.shape)
        if channels == 2:
            print("WARNING: stereo to mono")
            signal = signal[:, 0]

        sig_batch[i, :] = signal[snt_beg:snt_end] * rand_amp_arr[i]
        lab_batch[i] = int(label_list[snt_id_arr[i]][1])

    inp = flow.Tensor(sig_batch, dtype=flow.float32).to("cuda")
    lab = flow.Tensor(lab_batch, dtype=flow.float32).to("cuda")

    return inp, lab


if __name__ == "__main__":
    with open("data_preprocessed/label_dict.json", "r") as f:
        lab_dict = json.load(f)

    inp, lab = create_batches_rnd(
        lab_dict, batch_size=32, wlen=3200, fact_amp=0.2, train=True
    )
    print("inp: ", inp.size(), inp)
    print("lab: ", lab.size(), lab)
