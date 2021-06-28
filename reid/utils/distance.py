# -*- coding:utf-8 -*-
import numpy as np
from scipy.spatial.distance import cdist


def compute_distance_matrix(input1, input2, metric='euclidean'):
    """A wrapper function for computing distance matrix.

    Args:
        input1 (numpy.ndarray): 2-D feature matrix.
        input2 (numpy.ndarray): 2-D feature matrix.
        metric (str, optional): "euclidean" or "cosine".
            Default is "euclidean".

    Returns:
        numpy.ndarray: distance matrix.
    """
    # check input

    assert input1.shape[1] == input2.shape[1]

    if metric == 'euclidean':
        distmat = euclidean_squared_distance(input1, input2)
    elif metric == 'cosine':
        distmat = cosine_distance(input1, input2)
    else:
        raise ValueError(
            'Unknown distance metric: {}. '
            'Please choose either "euclidean" or "cosine"'.format(metric)
        )

    return distmat


def euclidean_squared_distance(input1, input2):
    dist = cdist(input1, input2, metric='euclidean').astype(np.float16)
    distmat = np.power(dist, 2).astype(np.float16)
    return distmat


def cosine_distance(input1, input2):
    dist = cdist(input1, input2, metric='cosine').astype(np.float16)
    distmat = np.power(dist, 2).astype(np.float16)
    return distmat
