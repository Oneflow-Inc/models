import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
import os
import argparse


RANDOM_SEED = 2018
cols = ["Label"]
for i in range(1, 14):
    cols.append("I" + str(i))
for i in range(1, 27):
    cols.append("C" + str(i))


def split_criteo(args):
    criteo_txt_path = args.criteo_txt_path
    ddf = pd.read_csv(
        criteo_txt_path, sep="\t", header=None, names=cols, encoding="utf-8", dtype=object
    )
    X = ddf.values
    y = ddf["Label"].map(lambda x: float(x)).values
    print(str(len(X)) + " lines in total")

    folds = StratifiedKFold(n_splits=10, shuffle=True, random_state=RANDOM_SEED).split(X, y)

    fold_indexes = []
    for train_id, valid_id in folds:
        fold_indexes.append(valid_id)
    test_index = fold_indexes[0]
    valid_index = fold_indexes[1]
    train_index = np.concatenate(fold_indexes[2:])

    test_df = ddf.loc[test_index, :]
    test_df.to_csv(os.path.join(args.output_data_dir, "test.csv"), index=False, encoding="utf-8")
    valid_df = ddf.loc[valid_index, :]
    valid_df.to_csv(os.path.join(args.output_data_dir, "valid.csv"), index=False, encoding="utf-8")
    ddf.loc[train_index, :].to_csv(
        os.path.join(args.output_data_dir, "train.csv"), index=False, encoding="utf-8"
    )

    print("Train lines:", len(train_index))
    print("Validation lines:", len(valid_index))
    print("Test lines:", len(test_index))
    print("Postive ratio:", np.sum(y) / len(y))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--criteo_txt_path", type=str, default="dac/train.txt", help="the raw Criteo data"
    )
    parser.add_argument(
        "--output_data_dir",
        type=str,
        default="../Criteo/Criteo_csv",
        help="the splited data output directory",
    )
    args = parser.parse_args()
    split_criteo(args)


