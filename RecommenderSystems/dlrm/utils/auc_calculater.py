import os
import sys
import time
import pickle
import numpy as np
from sklearn.metrics import roc_auc_score


def calculate_auc_from_file(pkl):
    results = pickle.load(open(pkl, 'rb'))
    labels = results['labels']
    preds = results['preds']
    iter = results['iter']
    labels = np.concatenate(labels, axis=0)
    preds = np.concatenate(preds, axis=0)
    auc = roc_auc_score(labels, preds)
    print('iter', iter, auc, time.time(), labels.shape[0])


def calculate_auc_from_dir(directory, startswith='eval_results_iter'):
    print('calculate AUC from folder:', directory)
    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        if filename.startswith(startswith) and filename.endswith(".pkl"): 
            calculate_auc_from_file(os.path.join(directory, filename))


if __name__ == "__main__":
    assert len(sys.argv) == 2, 'please input directory'
    calculate_auc_from_dir(sys.argv[1])
