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


def pose_orientation(array: np.ndarray,
                     rotation_vectors: np.ndarray,
                     plane_normals: np.ndarray,
                     origins: np.ndarray = None) -> np.ndarray:
    """Rotates all positions towards a plane that is defined by its normal
    vector and returns the rotated positions.

    The rotation will be computed based on the angle between the given
    rotation vectors and the plane normals (or actually the plane itself) so
    that each rotation vector is aligned to the defined plane. The resulting
    Rotations will then be applied about the given origins.

    Args:
        array (np.ndarray): The input array of dimensionality =3.
            Where the first dimension = number of frames,
            the second dimension = number of positions,
            and the last dimension = number of axis.
        rotation_vectors (np.ndarray): A point vector or an array of point
            vectors with size equal to number of frames.
        plane_normals (np.ndarray): A vector that defines the normal the plane
            where the array should be rotated to. Can either contain a
            3-d vector or an array of 3-d vectors with size = number of input
            frames.
        origins (np.ndarray): The point to rotate about. Can either
            contain 3-d point coordinates or a list of 3-d point coordinates
            with size = number of input frames.

    Returns:
        np.ndarray: The array rotated to the plane.

    Raises:
        ValueError: if input array is not of dimensionality = 3.
        ValueError: if array of rotation, plane normals or origins contain more
        than two dimensions (ndim > 2).
        ValueError: if array of rotation, plane normals or origins do not
        contain 3-D vectors/points or an array of 3-D vectors/points whose
        length equals the number of frames in array (array.shape[0]).

    """
    if not isinstance(origins, np.ndarray):
        if not origins:
            origins = np.array([0, 0, 0])

    n_frames = array.shape[0]

    if array.ndim != 3:
        raise ValueError('Input array must be of dimensionality = 3!')

    if rotation_vectors.ndim > 2:
        raise ValueError('Rotation vectors must be a 1 or 2 dimensional numpy '
                         'array')
    if rotation_vectors.shape[0] not in [3, n_frames]:
        raise ValueError('Rotation vectors must contain one vector or number '
                         'of frames times two vectors!')

    if plane_normals.ndim > 2:
        raise ValueError(
            'Plane normals must be a 1 or 2 dimensional numpy array')
    if plane_normals.shape[0] not in [3, n_frames]:
        raise ValueError('Plane normals must contain only one vector '
                         '(shape = 3) or as many vectors as frames.')

    if origins.ndim > 2:
        raise ValueError('Origins must be a 1 or 2 dimensional numpy array')
    if origins.shape[0] not in [3, n_frames]:
        raise ValueError('Origins must contain only one vector '
                         '(shape = 3) or as many vectors as frames.')

    if rotation_vectors.ndim == 1:
        rotation_vectors = np.expand_dims(rotation_vectors, 0)
    if plane_normals.ndim == 1:
        plane_normals = np.expand_dims(plane_normals, 0)
    if origins.ndim == 1:
        origins = np.expand_dims(origins, 0)

    # compute rotation angle based on rotation vectors and plane orthogonals
    alphas = mt.angle_complementary(rotation_vectors, plane_normals)

    # If more than one rotation vector is given but only one plane normal is
    # present, repeat to an array of (same) plane normals.
    if plane_normals.shape[0] == 1 and rotation_vectors.shape[0] > 1:
        plane_normals = np.full(rotation_vectors.shape, (plane_normals[0]))

    # If more than one plane normal is given but only one rotation vector is
    # present, repeat to an array of (same) rotation vectors.
    if rotation_vectors.shape[0] == 1 and plane_normals.shape[0] > 1:
        rotation_vectors = np.full(plane_normals.shape, (rotation_vectors[0]))

    # compute rotation axis from rotation vector and plane normals
    axes = np.abs(mt.orthogonal_vector(rotation_vectors, plane_normals))

    # create rotation matrix about axis
    rotations = mt.rotation(axes, alphas)

    # apply rotation matrix to vector and return
    # therefore we need to transpose the input arrays
    return np.transpose(rotations @ np.transpose(array, (0, 2, 1)), (0, 2, 1))
