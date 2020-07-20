import math
import numpy as np
import sklearn.preprocessing as preprocessing
from typing import Union


def norm(v: 'np.ndarray') -> 'np.ndarray':
    """Normalizes the given vector or array of vectors vec and returns it. Returns a numpy array that either represents the normalized vector or an array of normalized vectors.

    Args:
        vec_arr (np.ndarray): Either a 1-D array that represents the vector to normalize or a 2-D array containing the vectors to normalize.
    """
    if type(v) != np.ndarray:
        raise ValueError('v must be of type numpy.ndarray')
    if v.ndim > 2:
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
    if v1.ndim != v2.ndim or v1.ndim > 2:
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
    if v1.ndim != v2.ndim or v1.ndim > 2:
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
    if type(v) != np.ndarray:
        raise ValueError('v must be of type numpy.ndarray')
    if v.ndim > 2:
        raise ValueError('v must be either 1- or 2-dimensional (v.ndim == 1|2)')

    if v.ndim == 1:
        return np.append(v, 1)
    elif v.ndim == 2:
        return np.hstack((v, np.ones(len(v)).reshape((len(v), 1))))


def rotation(axis: 'np.ndarray', alpha) -> 'np.ndarray':
    # Source: https://stackoverflow.com/questions/6802577/rotation-of-3d-vector
    """Returns a 3x3 numpy array that represents a rotation about the given axis for the given angle alpha (radians)
        or Returns an array of rotations if axis and alpha are arrays of axes/angles.
    
    Args:
        axis (np.ndarray): A 3-D (ndim=3) vector that describes the axis to rotate about.
        alpha (float or np.ndarray): The angle to rotate for in radians.
    """
    if type(axis) != np.ndarray:
        raise ValueError('axis must be of type numpy.ndarray')
    if axis.ndim > 2:
        raise ValueError('v must be either 1- or 2-dimensional (v.ndim == 1|2)')
    if type(alpha) != float and type(alpha) != np.ndarray:
        raise ValueError('alpha must be of type float or numpy.ndarray')

    axis = norm(axis)
    a = np.cos(alpha / 2)
    if axis.ndim == 2 and type(alpha) == np.ndarray:
        # Transpose matrix as we want b,c,d being the x,y,z component
        # of each calculated vector from a list of axes
        b, c, d = (-axis.transpose() * np.sin(alpha / 2.0))
    else:
        # Just assign x,y,z components to b,c,d if ndim == 1
        b, c, d = (-axis * np.sin(alpha / 2.0))

    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    matrix = np.array([
        [aa + bb - cc - dd, 2.0 * (bc + ad), 2.0 * (bd - ac)],
        [2.0 * (bc - ad), aa + cc - bb - dd, 2.0 * (cd + ab)],
        [2.0 * (bd + ac), 2.0 * (cd - ab), aa + dd - bb - cc],
    ])
    # yapf: disable

    if axis.ndim == 2 and type(alpha) == np.ndarray:
        # Swap axes to get array of rotation-matrices instead of
        # 'matrix of matrices that include the rows of each rotation'
        matrix = matrix.swapaxes(1,2)
        matrix = matrix.swapaxes(0,1)
    return matrix


def transformation(rotation: 'np.ndarray', translation: 'np.ndarray') -> 'np.ndarray':
    """Returns a 4x4 transformation matrix including a rotation and a translation
        or an array of 4x4 transformation matrices.

    Args:
        rotation (np.ndarray): A 3x3 rotation matrix or an array of matrices. (default=np.array([[1,0,0],[0,1,0],[0,0,1]]))
        translation (np.ndarray): A 3-D translation vector or an array of vectors. (default=np.zeros(3))
    """
    if rotation is None:
        rotation = np.array([[1,0,0], [0,1,0], [0,0,1]])
    if translation is None:
        translation = np.array([0,0,0])

    if type(rotation) != np.ndarray or type(translation) != np.ndarray:
        raise ValueError('rotation and tranlastion must be of type numpy.ndarray')
    if rotation.ndim > 3:
        raise ValueError('rotation must be either 2- or 3-dimensional (v.ndim == 2|3)')
    if translation.ndim > 2:
        raise ValueError('translation must be either 1- or 2-dimensional (v.ndim == 1|2)')
    if rotation.ndim != translation.ndim + 1:
        raise ValueError('rotation (ndim=2|3) must have one more dimension than translation (ndim=1|2)')


    h = np.array([0, 0, 0, 1])
    if rotation.ndim == 2 and translation.ndim == 1:
        translation = translation.reshape(-1, 1)
        transformation = np.concatenate((rotation, translation), axis=1)
        # Add a dimension to h to be able to concatenate
        transformation = np.concatenate((transformation, h.reshape(1,-1)), axis=0)
        return transformation
    if rotation.ndim == 3 and translation.ndim == 2:
        translation = translation.reshape(len(translation), -1, 1)
        transformation = np.concatenate((rotation, translation), axis=2)
        # Add a dimension to h to be able to concatenate
        h = np.full((len(transformation), 1, 4), h.reshape(-1, 1, 4))
        transformation = np.concatenate((transformation, h), axis=1)
        return transformation

def rotation_from_vectors(v_from, v_to):
    """Returns a rotation matrix that rotates v_from so that it is aligned to v_to
        or returns an array of rotation matrices.

    Args:
        v_from (np.ndarray): A 3-D vector that defines the starting direction or an array of vectors.
        v_to (np.ndarray): A 3-D vector that defines the direction v_from points to after it has been rotated by the returned rotation or an array of vectors.
    """

    if v_from.shape != v_to.shape:
        raise ValueError('v_from and v_to have to share the same shapes')
    if v_from.ndim > 2 or v_to.ndim > 2:
        raise ValueError('Translation must be either 1- or 2-dimensional (v.ndim == 1|2)')

    v_from = norm(v_from)
    v_to = norm(v_to)
    alpha = get_angle(v_from, v_to)
    #TODO: Continue


def orthogonal_vector(v1: 'np.ndarray', v2: 'np.ndarray') -> 'np.ndarray':
    """Returns a vector that is perpendicular to v1 and v2
        or an array of vectors that is perpendicular to each pair of vectors in v1 and v2.

    Args:
        vec1 (np.ndarray): Vector one, which is perpendicular to the returned vector or an array of vectors.
        vec2 (np.ndarray): Vector two, which is perpendicular to the returned vector or an array of vectors.
    """

    if v1.shape != v2.shape:
        raise ValueError('v1 and v2 have to share the same shapes')
    if v1.ndim > 2 or v2.ndim > 2:
        raise ValueError('v1 and v2 must be either 1- or 2-dimensional (v.ndim == 1|2)')

    v1 = norm(v1)
    v2 = norm(v2)
    v1dotv2 = dot(v1, v2)

    if v1.ndim == 1 and v2.ndim == 1:
        # If dot product between v1 is -1/1 the vectors are parallel, so use any other unparallel vector
        if v1dotv2 == -1 or v1dotv2 == 1:
            return orthogonal_vector(np.random.rand(3), v2)
        else:
            return norm(np.cross(v1,v2))

    if v1.ndim == 2 and v2.ndim == 2:
        v_perpendicular = norm(np.cross(v1, v2))
        parallels = np.where((np.isclose(v1dotv2, -1)) | (np.isclose(v1dotv2, 1)))[0]
        # For all parallel v1/v2 pairs, reassign v1 with a random vector and check if its parallel, again.
        for idx in parallels:
            v_perpendicular[idx] = orthogonal_vector(np.random.rand(3), v2[idx])

        return v_perpendicular
