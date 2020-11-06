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
    return array - np.mean(array, axis=(0, 1))


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
                     v_from: np.ndarray,
                     v_to: np.ndarray,
                     axis: np.ndarray = None,
                     origin: np.ndarray = None) -> np.ndarray:
    """Rotates all positions around an axis based on the angle computed between
    v_from to v_to around axis, so that v_from is parallel to v_to.

    Args:
        array (np.ndarray): The input array of dimensionality =3.
            Where the first dimension = number of frames,
            the second dimension = number of positions,
            and the last dimension = number of axis.
        v_from (np.ndarray): The input vector to start rotation from. Can
            either be a 3d vector or an array of 3d vectors.
        v_to (np.ndarray): The target vector to rotate to. Can either be a 3d
            vector or an array of 3d vectors.
        axis (np.ndarray): The rotation axis. Can either be a 3d vector or an
            array of 3d vectors. Defaults to ([0, 0, 1]).
        origin (np.ndarray): The point to rotate about. Can either
            contain 3-d point coordinates or an array of 3-d point coordinates
            with size = number of input frames. Defaults to ([0, 0, 0]).

    Note: You can supply one or number of frames (array) distinct vectors for
        each parameter. If one vector is supplied it will be broadcasted. In
        case of number of frames vectors, those got applied to each frames
        seperately.

    Returns:
        np.ndarray: The rotated array around the axis.

    Raises:
        ValueError: if array is not of dimensionality equals 3.
        ValueError: if v_from is not of dimensionality equals 1 or 2.
        ValueError: if v_to is not of dimensionality equals 1 or 2.
        ValueError: if axis is not of dimensionality equals 1 or 2.
        ValueError: if origin is not of dimensionality equals 1 or 2.
    """
    if not isinstance(axis, np.ndarray):
        if not axis:
            axis = np.array([0, 0, 1])
    if not isinstance(origin, np.ndarray):
        if not origin:
            origin = np.array([0, 0, 0])

    n_frames = array.shape[0]

    if array.ndim != 3:
        raise ValueError('Input array must be of dimensionality = 3!')

    if v_from.ndim > 2:
        raise ValueError('v_from must be a 1 or 2 dimensional numpy array')
    if v_from.shape[0] not in [3, n_frames]:
        raise ValueError('v_from must contain one vector or number '
                         'of frames times two vectors!')

    if v_to.ndim > 2:
        raise ValueError('v_to must be a 1 or 2 dimensional numpy array')
    if v_to.shape[0] not in [3, n_frames]:
        raise ValueError('v_to must contain only one vector '
                         '(shape = 3) or as many vectors as frames.')

    if axis.ndim > 2:
        raise ValueError('axis must be a 1 or 2 dimensional numpy array')
    if axis.shape[0] not in [3, n_frames]:
        raise ValueError('axis must contain only one vector '
                         '(shape = 3) or as many vectors as frames.')

    if origin.ndim > 2:
        raise ValueError('Origin must be a 1 or 2 dimensional numpy array')
    if origin.shape[0] not in [3, n_frames]:
        raise ValueError('Origin must contain only one vector '
                         '(shape = 3) or as many vectors as frames.')

    if v_from.ndim == 1:
        v_from = np.expand_dims(v_from, 0)
    if v_to.ndim == 1:
        v_to = np.expand_dims(v_to, 0)
    if axis.ndim == 1:
        axis = np.expand_dims(axis, 0)
    if origin.ndim == 1:
        origin = np.expand_dims(origin, (0, 1))
    if origin.ndim == 2:
        origin = np.expand_dims(origin, (1))

    if v_from.shape != v_to:
        if v_from.shape[0] < v_to.shape[0]:
            v_from = np.full(v_to.shape, (v_from[0]))
        else:
            v_to = np.full(v_from.shape, (v_to[0]))

    # subtract origin from positions
    array = array - origin

    # TODO: remove repeat when orthogonal_vector method support broadcasting
    # If more than one input vector is given but only one axis is present,
    # repeat to an array of (same) axis.
    if axis.shape[0] == 1 and v_from.shape[0] > 1:
        axis = np.full(v_from.shape, (axis[0]))

    # compute cross product from vectors to rotation axis (plane normals)
    v_from_cross_axis = mt.orthogonal_vector(v_from, axis)
    v_to_cross_axis = mt.orthogonal_vector(axis, v_to)

    # compute angle between cross products
    alpha = mt.angle(v_from_cross_axis, v_to_cross_axis)

    # Compute new axis based on orthogonal from v_from_cross_axis and
    # v_to_cross_axis (may be negative or positive input axis).
    # This is necessary to define the correct rotation direction.
    # Only apply to values where angle != pi (180 degree)
    _filter = alpha != np.pi
    if len(_filter) == 1 and not _filter == False:
        axis[_filter] = mt.orthogonal_vector(v_from_cross_axis[_filter],
                                             v_to_cross_axis[_filter])

    # rotation matrix
    rotation = mt.rotation(axis, alpha)

    # apply rotation matrix to vector and return
    # therefore we need to transpose the input arrays
    # + (re)add the origin vector
    return np.transpose(rotation @ np.transpose(array, (0, 2, 1)),
                        (0, 2, 1)) + origin
