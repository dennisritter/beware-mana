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


def all_poses_mean(array, mode='all'):
    """Computes the mean of the frame(s) and subtracts it.

    There are two modes (single|all) where in *single* the mean will be
    computed for each frame and subtracted for each frame seperately.
    The mode *all* will compute a combined mean over all frames and
    subtract it for all frames respectively.
    The mean will be computed for each axis.

    Args:
        array (np.ndarray): The input array of abritary dimensionality >=2.
            Where the first dimension = number of frames,
            the second lst dimension = number of positions,
            and the last dimension = number of axis.
        mode (str): The mode on how to compute the mean (single|all). 
            Defaults to *all*.

    Returns:
        np.ndarray: The array subtracted by the mean of the first frame.
    """
    if array.ndim < 3:
        raise ValueError('Array need to have at least three dimensions!')

    if mode == 'single':
        return array - np.expand_dims(np.mean(array, axis=-2), -2)
    else:
        return array - np.mean(array, axis=(0, -2))


def pose_position(array: np.ndarray, position: np.ndarray) -> np.ndarray:
    """Normalizes the pose positions by translating them to a given origin.

    Args:
        array (np.ndarray): The input array of dimensionality =3.
            Where the first dimension = number of frames,
            the second lst dimension = number of positions,
            and the last dimension = number of axis.
        position (np.ndarray): The vector to translate all positions to.
            Number of axis must match the input array. Can be either a
            single array or multi-dimensional = number of input array frames.
            If multi-dimensional each frame will be subtracted by the
            corresponding position value.

    Returns:
        np.ndarray: The array translated to the specified origin.

    Raises:
        ValueError: if array is of dimensionalty < 3
        ValueErrof: if position dimensionality does not match number of input
            arry axis.
    """

    if not array.ndim == 3:
        raise ValueError('Array need to have three dimensions!')

    if position.shape[-1] == array.shape[-1]:
        return array - np.expand_dims(position, -2)
    else:
        raise ValueError(
            'Position array must match the input arrays number of axis!')
