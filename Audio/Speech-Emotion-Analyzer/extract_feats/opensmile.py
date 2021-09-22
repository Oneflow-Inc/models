import os
import csv
import sys
import time
from typing import Tuple, Union
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib
from sklearn.model_selection import train_test_split

# Number of features for each feature set
FEATURE_NUM = {
    "IS09_emotion": 384,
    "IS10_paraling": 1582,
    "IS11_speaker_state": 4368,
    "IS12_speaker_trait": 6125,
    "IS13_ComParE": 6373,
    "ComParE_2016": 6373,
}


def get_feature_opensmile(config, filepath: str) -> list:
    """
    Use Opensmile to extract (single) audio feature

    Args:
        config: configuration items
        file_path (str): path of the audio file

    Returns:
        vector (list): feature vector of this audio
    """

    # project path
    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
    # single_feature.csv 路径
    single_feat_path = os.path.join(BASE_DIR, config.feature_path, "single_feature.csv")
    # path of Opensmile site-packages
    opensmile_config_path = os.path.join(
        config.opensmile_path, "config", config.opensmile_config + ".conf"
    )
    print(opensmile_config_path)
    # Opensmile Command
    cmd = (
        "cd "
        + config.opensmile_path
        + " && ./SMILExtract -C "
        + opensmile_config_path
        + " -I "
        + filepath
        + " -O "
        + single_feat_path
    )
    print("Opensmile cmd: ", cmd)
    os.system(cmd)

    reader = csv.reader(open(single_feat_path, "r"))
    rows = [row for row in reader]
    last_line = rows[-1]
    return last_line[1 : FEATURE_NUM[config.opensmile_config] + 1]


def load_feature(
    config, feature_path: str, train: bool
) -> Union[Tuple[np.ndarray], np.ndarray]:
    """
    load feature data from `csv`

    Args:
        config: configuration
        feature_path (str): configuration items
        train (bool): training data

    Returns:
        - X (Tuple[np.ndarray]): training feature, testing fearture and labels
        - X (np.ndarray): predicting feature
    """

    #  load feature data
    df = pd.read_csv(feature_path)
    features = [str(i) for i in range(1, FEATURE_NUM[config.opensmile_config] + 1)]

    X = df.loc[:, features].values
    Y = df.loc[:, "label"].values

    # standardize the path of the model
    scaler_path = os.path.join(config.checkpoint_path, "SCALER_OPENSMILE.m")

    if train == True:
        # standardize the data
        scaler = StandardScaler().fit(X)
        # save
        joblib.dump(scaler, scaler_path)
        X = scaler.transform(X)

        # divide the training set and the test set
        x_train, x_test, y_train, y_test = train_test_split(
            X, Y, test_size=0.2, random_state=42
        )
        return x_train, x_test, y_train, y_test
    else:

        scaler = joblib.load(scaler_path)
        X = scaler.transform(X)
        return X


def get_data(
    config, data_path: str, feature_path: str, train: bool
) -> Union[Tuple[np.ndarray], np.ndarray]:
    """
    Extract all the audio features using Opensmile: go through all the folders, read the audio in each folder,
    extract the features of each audio, and save all the features to the path: `feature_path`
    Args:
        data_path (str): Dataset folder path
        feature_path (str): path of feature data
        train (bool): training data

    Returns:
        - train = True: training feature, testing fearture and labels
        - train = False: predicting feature
    """
    writer = csv.writer(open(feature_path, "w"))
    first_row = ["label"]
    for i in range(1, FEATURE_NUM[config.opensmile_config] + 1):
        first_row.append(str(i))
    writer.writerow(first_row)

    writer = csv.writer(open(feature_path, "a+"))
    print("Opensmile extracting...")

    if train == True:
        cur_dir = os.getcwd()
        sys.stderr.write("Curdir: %s\n" % cur_dir)
        os.chdir(data_path)
        # go through folder
        for i, directory in enumerate(config.class_labels):
            sys.stderr.write("Started reading folder %s\n" % directory)
            os.chdir(directory)

            # label_name = directory
            label = config.class_labels.index(directory)

            # read audio in this path
            for filename in os.listdir("."):
                if not filename.endswith("wav"):
                    continue
                filepath = os.path.join(os.getcwd(), filename)

                # extract feature
                feature_vector = get_feature_opensmile(config, filepath)
                feature_vector.insert(0, label)
                # write feature to csv file
                writer.writerow(feature_vector)

            sys.stderr.write("Ended reading folder %s\n" % directory)
            os.chdir("..")
        os.chdir(cur_dir)

    else:
        feature_vector = get_feature_opensmile(config, data_path)
        feature_vector.insert(0, "-1")
        writer.writerow(feature_vector)

    print("Opensmile extract done.")

    if train == True:
        return load_feature(config, feature_path, train=train)
