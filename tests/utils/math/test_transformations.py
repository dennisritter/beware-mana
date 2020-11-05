"""Unittests for transformation functions."""
import math

import numpy as np
import pytest

import mana.utils.math.transformations as t


def test_norm_vec_zero_vec():
    """Test whether a zero vector returns a zero vector."""
    zero_vector = np.zeros(3)
    result = t.norm_vec(zero_vector)
    assert np.array_equal(result,
                          zero_vector), 'Should return the same zero vector'


def test_norm_vec_unit_vec():
    """Test whether a unit vector returns a unit vector."""
    unit_vector = np.array([1, 0, 0])
    result = t.norm_vec(unit_vector)
    assert np.array_equal(result,
                          unit_vector), 'Should return the same unit vector'
    unit_vector = np.array([0, 1, 0])
    result = t.norm_vec(unit_vector)
    assert np.array_equal(result,
                          unit_vector), 'Should return the same unit vector'
    unit_vector = np.array([0, 0, 1])
    result = t.norm_vec(unit_vector)
    assert np.array_equal(result,
                          unit_vector), 'Should return the same unit vector'


def test_norm_vec_vec():
    """Test whether some vector is normalized correctly."""
    result = t.norm_vec(np.array([3, 4, 0]))
    assert np.array_equal(result, np.array([0.6, 0.8, 0.0]))


def test_norm_vec_batch():
    """Test whether an array of vectors is normalized correctly."""
    result = t.norm_vec(np.array([[0, 0, 0], [1, 0, 0], [3, 4, 0]]))
    assert np.array_equal(
        result, np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.6, 0.8, 0.0]]))


@pytest.mark.parametrize(
    'v1, v2, expected',
    [
        (np.array([1, 1, 1]), np.array([1, 1, 1]), 0.0),
        (np.array([1, 0, 0]), np.array([0, 1, 0]), math.pi / 2),
        (np.array([1, 0, 0]), np.array([1, 1, 0]), math.pi / 4),
    ],
)
def test_angle_single(v1, v2, expected):
    """Test whether a single angle is calculated correctly."""
    assert t.angle(v1, v2) == pytest.approx(expected)


@pytest.mark.parametrize(
    'v1, v2, expected',
    [
        (
            np.array([[1, 1, 1], [1, 0, 0], [1, 0, 0]]),
            np.array([[1, 1, 1], [0, 1, 0], [1, 1, 0]]),
            np.array([0.0, math.pi / 2, math.pi / 4]),
        ),
    ],
)
def test_angle_batch(v1, v2, expected):
    """Test whether multiple angles are calculated correctly at once."""
    assert t.angle(v1, v2) == pytest.approx(expected)


@pytest.mark.parametrize(
    'v1, v2, expected',
    [
        (np.array([1, 1, 1]), np.array([1, 1, 1]), math.cos(0.0)),
        (np.array([1, 0, 0]), np.array([0, 1, 0]), math.cos(math.pi / 2)),
        (np.array([1, 0, 0]), np.array([1, 1, 0]), math.cos(math.pi / 4)),
    ],
)
def test_dot_single(v1, v2, expected):
    """Test whether the dot product between a pair of vectors is calculated
    correctly.
    """
    assert t.dot(t.norm_vec(v1), t.norm_vec(v2)) == pytest.approx(expected)


@pytest.mark.parametrize(
    'v1, v2, expected',
    [
        (
            np.array([[1, 1, 1], [1, 0, 0], [1, 0, 0]]),
            np.array([[1, 1, 1], [0, 1, 0], [1, 1, 0]]),
            np.array(
                [math.cos(0.0),
                 math.cos(math.pi / 2),
                 math.cos(math.pi / 4)]),
        ),
    ],
)
def test_dot_batch(v1, v2, expected):
    """Test whether the dot product between multiple pairs of vectors is
    calculated correctly at once.
    """
    assert t.dot(t.norm_vec(v1), t.norm_vec(v2)) == pytest.approx(expected)


@pytest.mark.parametrize(
    'v, expected',
    [
        (np.array([1, 1, 1]), np.array([1, 1, 1, 1])),
        (np.array([1, 2, 3]), np.array([1, 2, 3, 1])),
    ],
)
def test_v3_to_v4_single(v, expected):
    """Test whether the homogenious coordinate (1) is appended to the given
    vector.
    """
    result = t.v3_to_v4(v)
    assert result == pytest.approx(expected)


@pytest.mark.parametrize(
    'v, expected',
    [
        (
            np.array([[1, 1, 1], [1, 2, 3]]),
            np.array([[1, 1, 1, 1], [1, 2, 3, 1]]),
        ),
    ],
)
def test_v3_to_v4_batch(v, expected):
    """Test whether the homogenious coordinate (1) is appended to all of the
    given vectors.
    """
    assert t.v3_to_v4(v) == pytest.approx(expected)


@pytest.mark.parametrize(
    'axis, alpha, expected',
    [
        (
            np.array([1.0, 0.0, 0.0]),
            math.pi / 2,
            np.array([
                [1.0, 0.0, 0.0],
                [0.0, np.cos(math.pi / 2), -np.sin(math.pi / 2)],
                [0.0, np.sin(math.pi / 2),
                 np.cos(math.pi / 2)],
            ]),
        ),
    ],
)
def test_rotation_single(axis, alpha, expected):
    """Test whether a rotation about the x-axis returns the correct rotation
    matrix.
    """
    result = t.rotation(axis, alpha)
    np.testing.assert_allclose(result, expected, atol=1e-07, verbose=True)


@pytest.mark.parametrize(
    'axis, alpha, expected',
    [
        (np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
        ]), np.array([
            math.pi / 2,
            math.pi / 4,
            math.pi / 8,
            math.pi / 10,
        ]),
         np.array([
             [
                 [1.0, 0.0, 0.0],
                 [0.0, np.cos(math.pi / 2), -np.sin(math.pi / 2)],
                 [0.0, np.sin(math.pi / 2),
                  np.cos(math.pi / 2)],
             ],
             [
                 [np.cos(math.pi / 4), 0.0,
                  np.sin(math.pi / 4)],
                 [0.0, 1.0, 0.0],
                 [-np.sin(math.pi / 4), 0.0,
                  np.cos(math.pi / 4)],
             ],
             [
                 [np.cos(math.pi / 8), -np.sin(math.pi / 8), 0.0],
                 [np.sin(math.pi / 8),
                  np.cos(math.pi / 8), 0.0],
                 [0.0, 0.0, 1.0],
             ],
             [
                 [np.cos(math.pi / 10), -np.sin(math.pi / 10), 0.0],
                 [np.sin(math.pi / 10),
                  np.cos(math.pi / 10), 0.0],
                 [0.0, 0.0, 1.0],
             ],
         ])),
        (np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]), np.array([
            math.pi / 2,
            math.pi / 4,
            math.pi / 8,
        ]),
         np.array([
             [
                 [1.0, 0.0, 0.0],
                 [0.0, np.cos(math.pi / 2), -np.sin(math.pi / 2)],
                 [0.0, np.sin(math.pi / 2),
                  np.cos(math.pi / 2)],
             ],
             [
                 [np.cos(math.pi / 4), 0.0,
                  np.sin(math.pi / 4)],
                 [0.0, 1.0, 0.0],
                 [-np.sin(math.pi / 4), 0.0,
                  np.cos(math.pi / 4)],
             ],
             [
                 [np.cos(math.pi / 8), -np.sin(math.pi / 8), 0.0],
                 [np.sin(math.pi / 8),
                  np.cos(math.pi / 8), 0.0],
                 [0.0, 0.0, 1.0],
             ],
         ])),
    ],
)
def test_rotation_batch(axis, alpha, expected):
    """Test whether multiple rotation matrices are calculated at once."""
    result = t.rotation(axis, alpha)
    np.testing.assert_allclose(result, expected, atol=1e-07, verbose=True)


@pytest.mark.parametrize(
    'rotation, translation, expected',
    [
        (
            np.array([
                [1.0, 0.0, 0.0],
                [0.0, np.cos(math.pi / 2), -np.sin(math.pi / 2)],
                [0.0, np.sin(math.pi / 2),
                 np.cos(math.pi / 2)],
            ]),
            np.array([1.0, 2.0, 3.0]),
            np.array([
                [1.0, 0.0, 0.0, 1.0],
                [0.0, np.cos(math.pi / 2), -np.sin(math.pi / 2), 2.0],
                [0.0, np.sin(math.pi / 2),
                 np.cos(math.pi / 2), 3.0],
                [0.0, 0.0, 0.0, 1.0],
            ]),
        ),
        (
            np.array([
                [1, 2, 3],
                [4, 5, 6],
                [7, 8, 9],
            ]),
            np.array([0, 0, 0]),
            np.array([
                [1, 2, 3, 0],
                [4, 5, 6, 0],
                [7, 8, 9, 0],
                [0, 0, 0, 1],
            ]),
        ),
        (
            np.array([
                [11, 22, 33],
                [44, 55, 66],
                [77, 88, 99],
            ]),
            np.array([6, 15, 24]),
            np.array([
                [11, 22, 33, 6],
                [44, 55, 66, 15],
                [77, 88, 99, 24],
                [0, 0, 0, 1],
            ]),
        ),
    ],
)
def test_transformation_single(rotation, translation, expected):
    """Test whether a transformation matrix is constructed correctly from given
    rotation and translation.
    """
    result = t.transformation(rotation, translation)
    np.testing.assert_array_equal(result, expected, verbose=True)


@pytest.mark.parametrize(
    'rotation, translation, expected',
    [
        (
            None,
            np.array([1, 2, 3]),
            np.array([
                [1.0, 0.0, 0.0, 1.0],
                [0.0, 1.0, 0.0, 2.0],
                [0.0, 0.0, 1.0, 3.0],
                [0, 0, 0, 1],
            ]),
        ),
    ],
)
def test_transformation_no_rotation(rotation, translation, expected):
    """Test whether a transformation matrix is constructed correctly from given
    translation but no rotation.
    """
    result = t.transformation(rotation, translation)
    np.testing.assert_array_equal(result, expected, verbose=True)


@pytest.mark.parametrize(
    'rotation, translation, expected',
    [
        (
            np.array([
                [1, 2, 3],
                [4, 5, 6],
                [7, 8, 9],
            ]),
            None,
            np.array([
                [1.0, 2.0, 3.0, 0.0],
                [4.0, 5.0, 6.0, 0.0],
                [7.0, 8.0, 9.0, 0.0],
                [0, 0, 0, 1],
            ]),
        ),
    ],
)
def test_transformation_no_translation(rotation, translation, expected):
    """Test whether a transformation matrix is constructed correctly from given
    rotation but no translation.
    """
    result = t.transformation(rotation, translation)
    np.testing.assert_array_equal(result, expected, verbose=True)


@pytest.mark.parametrize(
    'rotation, translation, expected',
    [
        (
            np.array([
                [
                    [1, 2, 3],
                    [4, 5, 6],
                    [7, 8, 9],
                ],
                [
                    [11, 22, 33],
                    [44, 55, 66],
                    [77, 88, 99],
                ],
            ]),
            np.array([
                [0, 0, 0],
                [6, 15, 24],
            ]),
            np.array([
                [
                    [1, 2, 3, 0],
                    [4, 5, 6, 0],
                    [7, 8, 9, 0],
                    [0, 0, 0, 1],
                ],
                [
                    [11, 22, 33, 6],
                    [44, 55, 66, 15],
                    [77, 88, 99, 24],
                    [0, 0, 0, 1],
                ],
            ]),
        ),
    ],
)
def test_transformation_batch(rotation, translation, expected):
    """Test whether multiple translations are constructed correctly at once."""
    result = t.transformation(rotation, translation)
    np.testing.assert_array_equal(result, expected, verbose=True)


@pytest.mark.parametrize(
    'rotation, translation, expected',
    [
        (
            np.array([
                [
                    [1, 2, 3],
                    [4, 5, 6],
                    [7, 8, 9],
                ],
                [
                    [11, 22, 33],
                    [44, 55, 66],
                    [77, 88, 99],
                ],
            ]),
            None,
            np.array([
                [
                    [1, 2, 3, 0],
                    [4, 5, 6, 0],
                    [7, 8, 9, 0],
                    [0, 0, 0, 1],
                ],
                [
                    [11, 22, 33, 0],
                    [44, 55, 66, 0],
                    [77, 88, 99, 0],
                    [0, 0, 0, 1],
                ],
            ]),
        ),
    ],
)
def test_transformation_batch_no_translation(rotation, translation, expected):
    """Test whether multiple translations are constructed correctly at once.
    Without a given translation"""
    result = t.transformation(rotation, translation)
    np.testing.assert_array_equal(result, expected, verbose=True)


@pytest.mark.parametrize(
    'rotation, translation, expected',
    [
        (
            np.array([
                [
                    [1, 2, 3],
                    [4, 5, 6],
                    [7, 8, 9],
                ],
                [
                    [11, 22, 33],
                    [44, 55, 66],
                    [77, 88, 99],
                ],
            ]),
            np.array([1, 2, 3]),
            np.array([
                [
                    [1, 2, 3, 1],
                    [4, 5, 6, 2],
                    [7, 8, 9, 3],
                    [0, 0, 0, 1],
                ],
                [
                    [11, 22, 33, 1],
                    [44, 55, 66, 2],
                    [77, 88, 99, 3],
                    [0, 0, 0, 1],
                ],
            ]),
        ),
    ],
)
def test_transformation_batch_one_translation_ndim1(rotation, translation,
                                                    expected):
    """Test whether multiple translations are constructed correctly at once.
    With one translation ndim=1 given. The translation should be broadcasted for
    all rotations."""
    result = t.transformation(rotation, translation)
    np.testing.assert_array_equal(result, expected, verbose=True)


@pytest.mark.parametrize(
    'rotation, translation, expected',
    [
        (
            np.array([
                [
                    [1, 2, 3],
                    [4, 5, 6],
                    [7, 8, 9],
                ],
                [
                    [11, 22, 33],
                    [44, 55, 66],
                    [77, 88, 99],
                ],
            ]),
            np.array([[1, 2, 3]]),
            np.array([
                [
                    [1, 2, 3, 1],
                    [4, 5, 6, 2],
                    [7, 8, 9, 3],
                    [0, 0, 0, 1],
                ],
                [
                    [11, 22, 33, 1],
                    [44, 55, 66, 2],
                    [77, 88, 99, 3],
                    [0, 0, 0, 1],
                ],
            ]),
        ),
    ],
)
def test_transformation_batch_one_translation_ndim2(rotation, translation,
                                                    expected):
    """Test whether multiple translations are constructed correctly at once.
    With one translation ndim=2 given. The translation should be broadcasted for
    all rotations."""
    result = t.transformation(rotation, translation)
    np.testing.assert_array_equal(result, expected, verbose=True)


@pytest.mark.parametrize(
    'rotation, translation, expected',
    [
        (
            np.array([
                [
                    [1, 2, 3],
                    [4, 5, 6],
                    [7, 8, 9],
                ],
            ]),
            np.array([
                [1, 2, 3],
                [6, 15, 24],
            ]),
            np.array([
                [
                    [1, 2, 3, 1],
                    [4, 5, 6, 2],
                    [7, 8, 9, 3],
                    [0, 0, 0, 1],
                ],
                [
                    [1, 2, 3, 6],
                    [4, 5, 6, 15],
                    [7, 8, 9, 24],
                    [0, 0, 0, 1],
                ],
            ]),
        ),
    ],
)
def test_transformation_batch_no_rotation(rotation, translation, expected):
    """Test whether multiple translations are constructed correctly at once.
    With one given rotation ndim=2. The rotation should be broadcasted for
    all translations."""
    result = t.transformation(rotation, translation)
    np.testing.assert_array_equal(result, expected, verbose=True)


@pytest.mark.parametrize(
    'rotation, translation, expected',
    [
        (
            np.array([
                [1, 2, 3],
                [4, 5, 6],
                [7, 8, 9],
            ], ),
            np.array([
                [1, 2, 3],
                [6, 15, 24],
            ]),
            np.array([
                [
                    [1, 2, 3, 1],
                    [4, 5, 6, 2],
                    [7, 8, 9, 3],
                    [0, 0, 0, 1],
                ],
                [
                    [1, 2, 3, 6],
                    [4, 5, 6, 15],
                    [7, 8, 9, 24],
                    [0, 0, 0, 1],
                ],
            ]),
        ),
    ],
)
def test_transformation_batch_one_rotation_ndim2(rotation, translation,
                                                 expected):
    """Test whether multiple translations are constructed correctly at once.
    With one given rotation ndim=2. The rotation should be broadcasted for
    all translations."""
    result = t.transformation(rotation, translation)
    np.testing.assert_array_equal(result, expected, verbose=True)


@pytest.mark.parametrize(
    'rotation, translation, expected',
    [
        (
            np.array([
                [
                    [1, 2, 3],
                    [4, 5, 6],
                    [7, 8, 9],
                ],
            ]),
            np.array([
                [1, 2, 3],
                [6, 15, 24],
            ]),
            np.array([
                [
                    [1, 2, 3, 1],
                    [4, 5, 6, 2],
                    [7, 8, 9, 3],
                    [0, 0, 0, 1],
                ],
                [
                    [1, 2, 3, 6],
                    [4, 5, 6, 15],
                    [7, 8, 9, 24],
                    [0, 0, 0, 1],
                ],
            ]),
        ),
    ],
)
def test_transformation_batch_one_rotation_ndim3(rotation, translation,
                                                 expected):
    """Test whether multiple translations are constructed correctly at once.
    With one given rotation ndim=3. The rotation should be broadcasted for
    all translations."""
    result = t.transformation(rotation, translation)
    np.testing.assert_array_equal(result, expected, verbose=True)


@pytest.mark.parametrize(
    'v1, v2, expected',
    [
        (
            np.array([1, 2, 3]),
            np.array([1, 1, 1]),
            np.array([-1.0, 2.0, -1.0]) /
            np.linalg.norm(np.array([-1.0, 2.0, -1.0])),
        ),
        (
            np.array([1, 1, 1]),
            np.array([1, 2, 3]),
            np.array([1.0, -2.0, 1.0]) /
            np.linalg.norm(np.array([1.0, -2.0, 1.0])),
        ),
    ],
)
def test_orthogonal_vector_single(v1, v2, expected):
    """Test whether an orthogonal vector is calculated correctly from two
    vectors.
    """
    result = t.orthogonal_vector(v1, v2)
    np.testing.assert_allclose(result, expected, atol=1e-07, verbose=True)


# For parallel vectors, don't assert the actual vector but whether the dot
# product between v1/v2 and the returned vector is 0 (90 degreees)
@pytest.mark.parametrize(
    'v1, v2, expected',
    [
        (np.array([1, 1, 1]), np.array([1, 1, 1]), 0),
        (np.array([1, 2, 3]), np.array([1, 2, 3]), 0),
    ],
)
def test_orthogonal_vector_single_parallel(v1, v2, expected):
    """Test whether the dot product between the returned vector and both given
    vectors is zero (90°).
    """
    result_vec = t.orthogonal_vector(v1, v2)
    assert t.dot(result_vec, v1) == pytest.approx(expected)
    assert t.dot(result_vec, v2) == pytest.approx(expected)


@pytest.mark.parametrize(
    'v1, v2, expected',
    [
        (
            np.array([[1, 1, 1], [1, 2, 3]]),
            np.array([[1, 2, 3], [1, 1, 1]]),
            np.array([
                [1.0, -2.0, 1.0] / np.linalg.norm(np.array([1.0, -2.0, 1.0])),
                [-1.0, 2.0, -1.0] / np.linalg.norm(np.array([-1.0, 2.0, -1.0]))
            ]),
        ),
    ],
)
def test_orthogonal_vector_multi(v1, v2, expected):
    """Test whether multiple orthogonal vectors are calculated correctly at
    once.
    """
    result = t.orthogonal_vector(v1, v2)
    np.testing.assert_allclose(result, expected, atol=1e-07, verbose=True)


# For parallel vectors, don't assert the actual vector but whether the dot
# product between v1/v2 and the returned vector is 0 (90 degreees)
@pytest.mark.parametrize(
    'v1, v2, expected',
    [
        (
            np.array([[1, 1, 1], [1, 2, 3]]),
            np.array([[1, 1, 1], [1, 2, 3]]),
            np.array([0.0, 0.0]),
        ),
    ],
)
def test_orthogonal_vector_multi_parallel(v1, v2, expected):
    """Test whether the dot product between returned vectors for multiple pairs
    of vectors are 0."""
    result = t.orthogonal_vector(v1, v2)
    np.testing.assert_allclose(t.dot(result, v1),
                               expected,
                               atol=1e-07,
                               verbose=True)
    np.testing.assert_allclose(t.dot(result, v2),
                               expected,
                               atol=1e-07,
                               verbose=True)


@pytest.mark.parametrize(
    'v_from, v_to, expected',
    [
        (
            np.array([1, 1, 1]),
            np.array([1, 2, 3]),
            np.array([1, 2, 3]),
        ),
        (
            np.array([1, 0, 0]),
            np.array([0, 1, 0]),
            np.array([0, 1, 0]),
        ),
    ],
)
def test_rotation_from_vectors_single(v_from, v_to, expected):
    """Test whether a rotation to transform a vector1 to be another vector2
    actually results in the vector2 when applied to vector1.
    """
    rotation = t.rotation_from_vectors(v_from, v_to)
    np.testing.assert_allclose(rotation @ t.norm_vec(v_from),
                               t.norm_vec(expected),
                               atol=1e-7,
                               verbose=True)


@pytest.mark.parametrize(
    'v_from, v_to, expected',
    [
        (
            np.array([1, 1, 1]),
            np.array([1, 1, 1]),
            np.identity(3),
        ),
        (
            np.array([1, 2, 3]),
            np.array([1, 2, 3]),
            np.identity(3),
        ),
    ],
)
def test_rotation_from_vectors_single_equals(v_from, v_to, expected):
    """Test whether the rotation matrix from two equal vectors is the identity
    matrix of same shape. (No rotation)
    """
    result = t.rotation_from_vectors(v_from, v_to)
    np.testing.assert_array_equal(result, expected)


@pytest.mark.parametrize(
    'v_from, v_to, expected',
    [
        (
            np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]]),
            np.array([[1, 2, 3], [1, 2, 3], [1, 2, 3]]),
            np.array([[1, 2, 3], [1, 2, 3], [1, 2, 3]]),
        ),
    ],
)
def test_rotation_from_vectors_multi(v_from, v_to, expected):
    """Test whether a rotation to transform a vector1 to be another vector2
    actually results in the vector2 when applied to vector1 for multiple pairs
    of vectors at once.
    """
    rotation = t.rotation_from_vectors(v_from, v_to)
    np.testing.assert_allclose(
        np.array([r @ t.norm_vec(v) for v, r in zip(v_from, rotation)]),
        t.norm_vec(expected),
        atol=1e-7,
        verbose=True,
    )


@pytest.mark.parametrize(
    'v_from, v_to, expected',
    [
        (
            np.array([[1, 1, 1], [1, 2, 3]]),
            np.array([[1, 1, 1], [1, 2, 3]]),
            np.array([np.identity(3), np.identity(3)]),
        ),
    ],
)
def test_rotation_from_vectors_multi_equals(v_from, v_to, expected):
    """Test whether the rotation matrices from multiple equal pairs of vectors
    are identity matrices of same shape. (No rotations)
    """
    result = t.rotation_from_vectors(v_from, v_to)
    np.testing.assert_array_equal(result, expected)


@pytest.mark.parametrize(
    'm1, m2, expected',
    [
        (
            np.array([
                [
                    [1, 2, 3],
                    [4, 5, 6],
                    [7, 8, 9],
                ],
                [
                    [1, 2, 3],
                    [4, 5, 6],
                    [7, 8, 9],
                ],
            ]),
            np.array([
                [
                    [1, 0, 0],
                    [0, 1, 0],
                    [0, 0, 1],
                ],
                [
                    [1, 2, 3],
                    [4, 5, 6],
                    [7, 8, 9],
                ],
            ]),
            np.array([
                [
                    [1, 2, 3],
                    [4, 5, 6],
                    [7, 8, 9],
                ],
                [
                    [30, 36, 42],
                    [66, 81, 96],
                    [102, 126, 150],
                ],
            ]),
        ),
    ],
)
def test_bmm_3x3(m1, m2, expected):
    """Test whether multiple 3x3 matrices are multipled correctly at once."""
    result = t.bmm(m1, m2)
    np.testing.assert_array_equal(result, expected)


@pytest.mark.parametrize(
    'm1, m2, expected',
    [
        (
            np.array([
                [
                    [1, 2, 3, 4],
                    [5, 6, 7, 8],
                    [9, 10, 11, 12],
                    [13, 14, 15, 16],
                ],
                [
                    [1, 2, 3, 4],
                    [5, 6, 7, 8],
                    [9, 10, 11, 12],
                    [13, 14, 15, 16],
                ],
            ]),
            np.array([
                [
                    [1, 0, 0, 0],
                    [0, 1, 0, 0],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1],
                ],
                [
                    [1, 0, 0, 0],
                    [0, 1, 0, 0],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1],
                ],
            ]),
            np.array([
                [
                    [1, 2, 3, 4],
                    [5, 6, 7, 8],
                    [9, 10, 11, 12],
                    [13, 14, 15, 16],
                ],
                [
                    [1, 2, 3, 4],
                    [5, 6, 7, 8],
                    [9, 10, 11, 12],
                    [13, 14, 15, 16],
                ],
            ]),
        ),
    ],
)
def test_bmm_4x4(m1, m2, expected):
    """Test whether multiple 4x4 matrices are multipled correctly at once."""
    result = t.bmm(m1, m2)
    np.testing.assert_array_equal(result, expected)


@pytest.mark.parametrize(
    'm, v, expected',
    [
        (np.array([[
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9],
        ], [
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9],
        ]]), np.array([
            [1, 1, 1],
            [1, 2, 3],
        ]), np.array([
            [6, 15, 24],
            [14, 32, 50],
        ])),
    ],
)
def test_bmvm_3x3(m, v, expected):
    """Test whether multiple 3x3 matrices are multipled with 3d-vectors
    correctly at once.
    """
    result = t.bmvm(m, v)
    np.testing.assert_array_equal(result, expected)


@pytest.mark.parametrize(
    'm, v, expected',
    [
        (np.array([[
            [1, 2, 3, 0],
            [4, 5, 6, 0],
            [7, 8, 9, 0],
            [0, 0, 0, 1],
        ], [
            [1, 2, 3, 0],
            [4, 5, 6, 0],
            [7, 8, 9, 0],
            [0, 0, 0, 1],
        ]]), np.array([
            [1, 1, 1, 1],
            [1, 2, 3, 1],
        ]), np.array([
            [6, 15, 24, 1],
            [14, 32, 50, 1],
        ])),
    ],
)
def test_bmvm_4x4(m, v, expected):
    """Test whether multiple 4x4 matrices are multipled with 3d-vectors
    correctly at once.
    """
    result = t.bmvm(m, v)
    np.testing.assert_array_equal(result, expected)


@pytest.mark.parametrize(
    'origin, x_dir, y_dir, expected',
    [
        (
            np.array([0, 0, 0]),
            np.array([1, 0, 0]),
            np.array([0, 1, 0]),
            np.identity(4),
        ),
    ],
)
def test_projection_matrix_(origin, x_dir, y_dir, expected):
    """Test whether the projection matrix of a child coordinate system that is
    already aligned to the parent coordinate system is the unit matrix.
    """
    result = t.projection_matrix(origin, x_dir, y_dir)
    np.testing.assert_allclose(result, expected)
