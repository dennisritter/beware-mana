import math
import numpy as np
import sklearn.preprocessing as preprocessing


def norm(vec_arr: 'np.ndarray') -> 'np.ndarray':
    """Normalizes the given vector or array of vectors vec and returns it. Returns a numpy array that either represents the normalized vector or an array of normalized vectors.

    Args:
        vec_arr (np.ndarray): Either a 1-D array that represents the vector to normalize or a 2-D array containing the vectors to normalize.
    """
    if vec_arr.ndim == 1:
        vec_norm = np.linalg.norm(vec_arr)

        # No zero division
        if vec_norm == 0:
            return np.zeros(len(vec_arr))
        return vec_arr / vec_norm
    if vec_arr.ndim == 2:
        return preprocessing.normalize(vec_arr, norm='l2')