"""Contains the normalization methods for transforming positions in space."""
import numpy as np


def first_pose_mean(array: np.ndarray) -> np.ndarray:
    """Computes the mean of the first frame and subtracts this from all frames.
    The mean will be computed for each axis.

    Args:
        array (np.ndarray): The input array of abritary dimensionality >=2.
            Where the first dimension = number of frames,
            the second lst dimension = number of positions,
            and the last dimension = number of axis.

    Returns:
        np.ndarray: The array subtracted by the mean of the first frame.
    """
    if array.ndim < 3:
        raise ValueError('Array need to have at least three dimensions!')
    mean_slice = [0] * (array.ndim - 2) + [slice(None), slice(None)]
    return array - np.mean(array[mean_slice], axis=0)


