import numpy as np
import pandas as pd
import argparse
from sklearn.model_selection import StratifiedKFold

RANDOM_SEED = 2018  # Fix seed for reproduction


def split_train_val_test(input_dir, output_dir):
    num_dense_fields = 13
    num_sparse_fields = 26

    fields = ["Label"]
    fields += [f"I{i+1}" for i in range(num_dense_fields)]
    fields += [f"C{i+1}" for i in range(num_sparse_fields)]

    ddf = pd.read_csv(
        f"{input_dir}/train.txt",
        sep="\t",
        header=None,
        names=fields,
        encoding="utf-8",
        dtype=object,
    )
    X = ddf.values
    y = ddf["Label"].map(lambda x: float(x)).values
    print(f"{len(X)} samples in total")

    folds = StratifiedKFold(n_splits=10, shuffle=True, random_state=RANDOM_SEED)

    fold_indexes = [valid_idx for _, valid_idx in folds.split(X, y)]
    test_index = fold_indexes[0]
    valid_index = fold_indexes[1]
    train_index = np.concatenate(fold_indexes[2:])

    ddf.loc[test_index, :].to_csv(
        f"{output_dir}/test.csv", index=False, encoding="utf-8"
    )
    ddf.loc[valid_index, :].to_csv(
        f"{output_dir}/valid.csv", index=False, encoding="utf-8"
    )
    ddf.loc[train_index, :].to_csv(
        f"{output_dir}/train.csv", index=False, encoding="utf-8"
    )

    print("Train lines:", len(train_index))
    print("Validation lines:", len(valid_index))
    print("Test lines:", len(test_index))
    print("Postive ratio:", np.sum(y) / len(y))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Path to downloaded criteo kaggle dataset",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Path to splitted criteo kaggle dataset",
    )
    args = parser.parse_args()
    split_train_val_test(args.input_dir, args.output_dir)
