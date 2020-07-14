import math
import numpy as np
import sklearn.preprocessing as preprocessing


def norm(v: 'np.ndarray') -> 'np.ndarray':
    """Normalizes the given vector or array of vectors vec and returns it. Returns a numpy array that either represents the normalized vector or an array of normalized vectors.

    Args:
        vec_arr (np.ndarray): Either a 1-D array that represents the vector to normalize or a 2-D array containing the vectors to normalize.
    """
    if type(v) != np.ndarray:
        raise ValueError('v must be of type numpy.ndarray')
    elif v.ndim > 2:
        raise ValueError('v must be either 1- or 2-dimensional')

    if v.ndim == 1:
        vec_norm = np.linalg.norm(v)

        # No zero division
        if vec_norm == 0:
            return np.zeros(len(v))
        return v / vec_norm
    if v.ndim == 2:
        return preprocessing.normalize(v, norm='l2')


def get_angle(v1, v2):
    """Returns the angle (radians) between the vectors v1, v2 if single vectors are given (v1,v2 ndim == 1).
       Returns an array of angles (radians) between respective vectors in v1, v2 arrays of vectors ((v1,v2 ndim == 2)).

    Args:
        v1 (np.ndarray): A 3-D vector or an array of 3-D vectors.
        v2 (np.ndarray): A 3-D vector or an array of 3-D vectors.
    """
    if type(v1) != np.ndarray or type(v2) != np.ndarray:
        raise ValueError('v1 and v2 must both be of type numpy.ndarray')
    elif v1.ndim != v2.ndim or v1.ndim > 2:
        raise ValueError('v1 and v2 must both be either 1- or 2-dimensional')
    # * The dot product of two equal vectors will result in a minor rounding error (dot(norm([1,1,1])) = dot([0.57735027, 0.57735027, 0.57735027]) = 1.0000000000000002)
    # * As np.arccos can only handle values from 0.0 to 1.0 (inclusive), we clamp the result to this range, too.
    return np.arccos(np.clip(dot(norm(v1), norm(v2)), 0.0, 1.0))


def dot(v1, v2):
    """Returns the dot product between v1 and v2 (v1,v2 ndim == 1).
       Returns an array of dot products between respective vectors in v1 and v2 (v1,v2 ndim == 2).

    Args:
        v1 (np.ndarray): A 3-D vector or an array of 3-D vectors.
        v2 (np.ndarray): A 3-D vector or an array of 3-D vectors.
    """
    if type(v1) != np.ndarray or type(v2) != np.ndarray:
        raise ValueError('v1 and v2 must both be of type numpy.ndarray')
    elif v1.ndim != v2.ndim or v1.ndim > 2:
        raise ValueError('v1 and v2 must both be either 1- or 2-dimensional')

    if v1.ndim == 1 and v2.ndim == 1:
        return np.dot(v1, v2)
    elif v1.ndim == 2 and v2.ndim == 2:
        return (v1 @ v2.transpose()).diagonal()


def v3_to_v4(v):
    """Returns a 3-D position vector with appended homogenious coordinate from the given 3-D vector or an array of such vectors.

    Args:
        v (np.ndarray): A 3-D vector or an array of 3-D vectors.
    """
    if type(v) != np.ndarray or type(v) != np.ndarray:
        raise ValueError('v must be of type numpy.ndarray')
    elif v.ndim > 2:
        raise ValueError('v must be either 1- or 2-dimensional (v.ndim == 1|2)')

    if v.ndim == 1:
        return np.append(v, 1)
    elif v.ndim == 2:
        return np.hstack((v, np.ones(len(v)).reshape((len(v), 1))))