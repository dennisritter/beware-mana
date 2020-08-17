"""Contains the normalization methods for transforming positions in space."""
import numpy as np

import mana.utils.math.transformations as mt


def pose_mean(array: np.ndarray, mean_array: np.ndarray = None) -> np.ndarray:
    """Computes the mean of the frame(s) and subtracts it.

    The mean will be computed for each axis.

    Args:
        array (np.ndarray): The input array of abritary dimensionality = 3.
            Where the first dimension = number of frames,
            the second dimension = number of positions,
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
            the second dimension = number of positions,
            and the last dimension = number of axis.
        position (np.ndarray): The vector to translate all positions to.
            Number of axis must match the input array. Can be either a
            single array or multi-dimensional = number of input array frames.
            If multi-dimensional each frame will be subtracted by the
            corresponding position value.

    Returns:
        np.ndarray: The array translated to the specified origin.

    Raises:
        ValueError: if array is not of dimensionalty = 3.
        ValueErrof: if position dimensionality does not match number of input
            arry axis.
    """

    if not array.ndim == 3:
        raise ValueError('Array need to have three dimensions!')

    if not position.shape[-1] == array.shape[-1]:
        raise ValueError(
            'Position array must match the input arrays number of axis!')

    return array - np.expand_dims(position, -2)


def pose_orientation(array: np.ndarray, rotation_vectors: np.ndarray,
                     orthogonal_vectors: np.ndarray,
                     origin_vectors: np.ndarray) -> np.ndarray:
    """Rotates all positions towards a defined plane.

    The rotation will be computed based on the given rotation point vectors, by
    subtracting one from the other. The rotation angle results based on this
    new vector compared to the orthogonal vector. Rotation will then be applied
    around the given origin vector.

    Args:
        array (np.ndarray): The input array of dimensionality =3.
            Where the first dimension = number of frames,
            the second dimension = number of positions,
            and the last dimension = number of axis.
        rotation_vectors (np.ndarray): Array consisting of two point vectors.
            Array can either contain two vectors or a list of two vectors with
            size = number of input frames. The first vector will be subtracted
            by the second.
        orthogonal_vectors (np.ndarray): The orthogonal vector of the plane
            where the array should be rotated to. Can either contain a
            3-dimensional vector or a list of 3-d vectors with size = number of
            input frames.
        origin_vectors (np.ndarray): The vector to rotate around. Can either
            contain a 3-dimensional vector or a list of 3-d vectors with size
            = number of input frames.

    Returns:
        np.ndarray: The array rotated to the plane.

    Raises:
        ValueError: if input array is not of dimensionality = 3.
        ValueError: if list of rotation, orthogonal or origin vectors are not
            of size 1 or of size equal to input array frames.
    """
    if array.ndim != 3:
        raise ValueError('Input array must be of dimensionality = 3!')
    frames = array.shape[0]

    if rotation_vectors.ndim != 2 or rotation_vectors.ndim != 3:
        raise ValueError('Rotation vectors must be a 2 or 3 dimensional numpy '
                         'array')
    if rotation_vectors.shape[0] not in [1, frames]:
        raise ValueError('Rotation vectors must contain two vectors or number \
                          of frames times two vectors!')
    if rotation_vectors.ndim == 2:
        rotation_vectors = np.expand_dims(rotation_vectors, 0)

    if orthogonal_vectors.ndim != 1 or orthogonal_vectors.ndim != 2:
        raise ValueError(
            'Orthogonal vectors must be a 1 or 2 dimensional numpy array')
    if orthogonal_vectors.shape[0] not in [1, frames]:
        raise ValueError('Orthogonal vectors must contain only one vector or \
                          number of frames times one vector!')
    if orthogonal_vectors.ndim == 1:
        orthogonal_vectors = np.expand_dims(orthogonal_vectors, 0)

    if origin_vectors.ndim != 1 or orthogonal_vectors.ndim != 2:
        raise ValueError(
            'Origin vectors must be a 1 or 2 dimensional numpy array')
    if origin_vectors.shape[0] not in [1, frames]:
        raise ValueError('Origin vectors must contain only one vector or \
                          number of frames times one vector!')
    if origin_vectors.ndim == 1:
        origin_vectors = np.expand_dims(origin_vectors, 0)

    # compute pose vectors as difference between rotation vectors
    pose_vectors = rotation_vectors[:, 0] - rotation_vectors[:, 1]

    # compute rotation angle based on pose vectors and plane orthogonals
    alphas = mt.angle_complementary(pose_vectors, orthogonal_vectors)

    # compute rotation axis from pose vector and orthogonals
    # TODO: ref to vectorization if orthogonal vector supports it
    # axes = np.abs(mt.orthogonal_vector(pose_vectors, orthogonal_vectors))
    axes = np.array([
        np.abs(mt.orthogonal_vector(pose_vector, orthogonal))
        for pose_vector, orthogonal in zip(pose_vectors, orthogonal_vectors)
    ])

    # create rotation matrix about axis
    rotations = mt.rotation(axes, alphas)

    # apply rotation matrix to vector and return
    # therefore we need to transpose the input arrays
    return np.transpose(rotations @ np.transpose(array, (0, 2, 1)), (0, 2, 1))
