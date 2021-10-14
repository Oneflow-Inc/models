import os
import time
import argparse
import pickle

import numpy as np

import data_utils as preprocess


def save_pickle(variable, fileName):
    with open(fileName, 'wb') as f:
        pickle.dump(variable, f)


def load_pickle_file(fileName):
    with open(fileName, 'rb') as f:
        return pickle.load(f)


def preprocess_for_training(data_path, speaker_id, cache_folder):
    num_mcep = 80
    sampling_rate = 22050
    frame_period = 5.0
    n_frames = 128

    print(f"Preprocessing data for speaker: {speaker_id}.")
    start_time = time.time()

    wavs = preprocess.load_wavs(wav_dir=data_path, sr=sampling_rate)

    f0s, timeaxes, sps, aps, coded_sps = preprocess.world_encode_data(
        wave=wavs, fs=sampling_rate, frame_period=frame_period, coded_dim=num_mcep)

    coded_sps_transposed = preprocess.transpose_in_list(lst=coded_sps)
    coded_sps_norm, coded_sps_mean, coded_sps_std = preprocess.coded_sps_normalization_fit_transform(
        coded_sps=coded_sps_transposed)

    if not os.path.exists(os.path.join(cache_folder, speaker_id)):
        os.makedirs(os.path.join(cache_folder, speaker_id))

    np.savez(os.path.join(cache_folder, speaker_id, f"{speaker_id}_norm_stat.npz"),
             mean=coded_sps_mean,
             std=coded_sps_std)

    save_pickle(variable=coded_sps_norm,
                fileName=os.path.join(cache_folder, speaker_id, f"{speaker_id}_normalized.pickle"))

    end_time = time.time()
    print("Preprocessing data for speaker: {speaker_id} finsihed!")

    print("Time taken for preprocessing {:.4f} seconds".format(
        end_time - start_time))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--data_directory', type=str, default='vcc2018/vcc2018_training',
                        help='Directory holding VCC2018 dataset.')
    parser.add_argument('--preprocessed_data_directory', type=str, default='vcc2018_preprocessed/vcc2018_training',
                        help='Directory holding preprocessed VCC2018 dataset.')
    parser.add_argument('--speaker_ids', nargs='+', type=str, default=['VCC2SM3', 'VCC2TF1'],
                        help='Source speaker id from VCC2018.')

    args = parser.parse_args()

    for speaker_id in args.speaker_ids:
        data_path = os.path.join(args.data_directory, speaker_id)
        preprocess_for_training(data_path=data_path, speaker_id=speaker_id,
                                cache_folder=args.preprocessed_data_directory)
