"""Contains the normalization methods for transforming positions in space."""
import numpy as np


def pose_mean(array: np.ndarray, mean_array: np.ndarray = None) -> np.ndarray:
    """Computes the mean of the frame(s) and subtracts it.

    The mean will be computed for each axis.

    Args:
        array (np.ndarray): The input array of abritary dimensionality = 3.
            Where the first dimension = number of frames,
            the second lst dimension = number of positions,
            and the last dimension = number of axis.
        mean_array (np.ndarray): The vector to compute the mean and subtract it
            from the input array. Number of axis must match the input array.
            Can be either a single array or multi-dimensional = number of input
            array frames. If multi-dimensional each frame will be subtracted by
            the corresponding mean value. Defaults to None = compute mean from
            first frame.

    Returns:
        np.ndarray: The array subtracted by the mean of the first frame.

    Raises:
        ValueError: if array is not of dimensionalty 3
        ValueErrof: if position dimensionality does not match number of input
            arry axis.
    """
    if array.ndim < 3:
        raise ValueError('Array need to have at least three dimensions!')

    if mean_array is not None and mean_array.shape[-1] != array.shape[-1]:
        raise ValueError(
            'Position array must match the input arrays number of axis!')
    return array - np.mean(array, axis=0)


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
        ValueError: if array is not of dimensionalty = 3
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
