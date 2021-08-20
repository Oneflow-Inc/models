import os
import json

import soundfile as sf


data_path = "data"
train_data_path = os.path.join(data_path, "train_data")
test_data_path = os.path.join(data_path, "test_data")
data_preprocessed_path = "data_preprocessed"
train_data_preprocessed_path = os.path.join(data_preprocessed_path, "train")
test_data_preprocessed_path = os.path.join(data_preprocessed_path, "test")
label_path = os.path.join(data_preprocessed_path, "label_dict.json")


def data_preprocess(data_dir, data_preprocessed_dir):
    data_list = []

    for file in os.listdir(data_dir):
        file_path = os.path.join(data_dir, file)

        if os.path.splitext(file_path)[1] == ".wav":
            [waveform, fs] = sf.read(file_path)
            waveform = waveform.T.mean(axis=0)
            txt_path = os.path.splitext(file_path)[0] + ".txt"
            assert txt_path.split("/")[-1] in os.listdir(
                data_dir
            ), "txt file is not exist."

            with open(txt_path, "r") as f:
                count = 0
                for line in f.readlines():
                    if len(line.strip().split("_")) == 4:
                        id, start, end, text = line.strip().split("_")
                        wave_snip = waveform[
                            int(eval(start) * fs) : int(eval(end) * fs)
                        ]
                        filename = os.path.splitext(file)[0] + "_" + str(count) + ".wav"

                        sf.write(
                            os.path.join(data_preprocessed_dir, filename), wave_snip, fs
                        )
                        data_list.append(
                            [os.path.join(data_preprocessed_dir, filename), id]
                        )
                        count += 1
                    else:
                        continue
    return data_list


def main():
    train_data_list = data_preprocess(train_data_path, train_data_preprocessed_path)
    test_data_list = data_preprocess(test_data_path, test_data_preprocessed_path)

    label_dict = {"train": train_data_list, "test": test_data_list}

    with open(label_path, "w") as f:
        json.dump(label_dict, f)


if __name__ == "__main__":
    main()
