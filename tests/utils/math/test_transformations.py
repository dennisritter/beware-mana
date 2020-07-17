import pytest
import mana.utils.math.transformations as t
import numpy as np
import math


# norm
def test_norm_zero_vec():
    zero_vector = np.zeros(3)
    result = t.norm(zero_vector)
    assert np.array_equal(result, zero_vector), 'Should return the same zero vector'


def test_norm_unit_vec():
    unit_vector = np.array([1, 0, 0])
    result = t.norm(unit_vector)
    assert np.array_equal(result, unit_vector), 'Should return the same unit vector'
    unit_vector = np.array([0, 1, 0])
    result = t.norm(unit_vector)
    assert np.array_equal(result, unit_vector), 'Should return the same unit vector'
    unit_vector = np.array([0, 0, 1])
    result = t.norm(unit_vector)
    assert np.array_equal(result, unit_vector), 'Should return the same unit vector'


def test_norm_vec():
    result = t.norm(np.array([3, 4, 0]))
    assert np.array_equal(result, np.array([0.6, 0.8, 0.0]))


def test_norm_batch():
    result = t.norm(np.array([[0, 0, 0], [1, 0, 0], [3, 4, 0]]))
    assert np.array_equal(result, np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.6, 0.8, 0.0]]))


# get_angle
@pytest.mark.parametrize('v1, v2, expected', [(np.array([1, 1, 1]), np.array([1, 1, 1]), 0.0), (np.array([1, 0, 0]), np.array([0, 1, 0]), math.pi / 2),
                                              (np.array([1, 0, 0]), np.array([1, 1, 0]), math.pi / 4)])
def test_get_angle_single(v1, v2, expected):
    assert t.get_angle(v1, v2) == pytest.approx(expected)


@pytest.mark.parametrize('v1, v2, expected',
                         [(np.array([[1, 1, 1], [1, 0, 0], [1, 0, 0]]), np.array([[1, 1, 1], [0, 1, 0], [1, 1, 0]]), np.array([0.0, math.pi / 2, math.pi / 4]))]
                         )
def test_get_angle_batch(v1, v2, expected):
    result = t.get_angle(v1, v2)
    # Can't assert with pyterst.approx in array of vectors
    assert result[0] == pytest.approx(expected[0])
    assert result[1] == pytest.approx(expected[1])
    assert result[2] == pytest.approx(expected[2])


# dot
@pytest.mark.parametrize('v1, v2, expected', [(np.array([1, 1, 1]), np.array([1, 1, 1]), math.cos(0.0)),
                                              (np.array([1, 0, 0]), np.array([0, 1, 0]), math.cos(math.pi / 2)),
                                              (np.array([1, 0, 0]), np.array([1, 1, 0]), math.cos(math.pi / 4))])
def test_dot_single(v1, v2, expected):
    assert t.dot(t.norm(v1), t.norm(v2)) == pytest.approx(expected)


@pytest.mark.parametrize('v1, v2, expected', [(np.array([[1, 1, 1], [1, 0, 0], [1, 0, 0]]), np.array(
    [[1, 1, 1], [0, 1, 0], [1, 1, 0]]), np.array([math.cos(0.0), math.cos(math.pi / 2), math.cos(math.pi / 4)]))])
def test_dot_batch(v1, v2, expected):
    # Can't assert with pyterst.approx in array of vectors
    result = t.dot(t.norm(v1), t.norm(v2))
    assert result[0] == pytest.approx(expected[0])
    assert result[1] == pytest.approx(expected[1])
    assert result[2] == pytest.approx(expected[2])


# v3_to_v4
@pytest.mark.parametrize('v, expected', [(np.array([1, 1, 1]), 4)])
def test_v3_to_v4_single(v, expected):
    result = len(t.v3_to_v4(v))
    assert result == pytest.approx(expected)


@pytest.mark.parametrize('v, expected', [(np.array([[1, 1, 1], [1, 2, 3]]), 4)])
def test_v3_to_v4_batch(v, expected):
    assert t.v3_to_v4(v).shape[1] == expected


@pytest.mark.parametrize(
    'axis, alpha, expected',
    [(np.array([1.0, 0.0, 0.0]), math.pi / 2,
      np.array([[1.0, 0.0, 0.0], [0.0, np.cos(math.pi / 2), -np.sin(math.pi / 2)], [0.0, np.sin(math.pi / 2), np.cos(math.pi / 2)]]))])
def test_rotation_single(axis, alpha, expected):
    result = t.rotation(axis, alpha)
    np.testing.assert_allclose(result, expected, atol=1e-07, verbose=True)


@pytest.mark.parametrize(
    'axis, alpha, expected',
    [(np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 1.0]]), np.array([math.pi / 2, math.pi / 4, math.pi / 8, math.pi / 10]),
      np.array([[[1.0, 0.0, 0.0], [0.0, np.cos(math.pi / 2), -np.sin(math.pi / 2)], [0.0, np.sin(math.pi / 2), np.cos(math.pi / 2)]],
                [[np.cos(math.pi / 4), 0.0, np.sin(math.pi / 4)], [0.0, 1.0, 0.0], [-np.sin(math.pi / 4), 0.0, np.cos(math.pi / 4)]],
                [[np.cos(math.pi / 8), -np.sin(math.pi / 8), 0.0], [np.sin(math.pi / 8), np.cos(math.pi / 8), 0.0], [0.0, 0.0, 1.0]],
                [[np.cos(math.pi / 10), -np.sin(math.pi / 10), 0.0], [np.sin(math.pi / 10), np.cos(math.pi / 10), 0.0], [0.0, 0.0, 1.0]]]))])
def test_rotation_batch_mxn(axis, alpha, expected):
    result = t.rotation(axis, alpha)
    np.testing.assert_allclose(result, expected, atol=1e-07, verbose=True)


@pytest.mark.parametrize(
    'axis, alpha, expected',
    [(np.array([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]), np.array([math.pi / 2, math.pi / 2, math.pi / 2]),
      np.array([[[1.0, 0.0, 0.0], [0.0, np.cos(math.pi / 2), -np.sin(math.pi / 2)], [0.0, np.sin(math.pi / 2), np.cos(math.pi / 2)]],
                [[1.0, 0.0, 0.0], [0.0, np.cos(math.pi / 2), -np.sin(math.pi / 2)], [0.0, np.sin(math.pi / 2), np.cos(math.pi / 2)]],
                [[1.0, 0.0, 0.0], [0.0, np.cos(math.pi / 2), -np.sin(math.pi / 2)], [0.0, np.sin(math.pi / 2), np.cos(math.pi / 2)]]]))])
def test_rotation_batch_nxn_debug(axis, alpha, expected):
    result = t.rotation(axis, alpha)
    np.testing.assert_allclose(result, expected, atol=1e-07, verbose=True)


@pytest.mark.parametrize(
    'axis, alpha, expected',
    [(np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]), np.array([math.pi / 2, math.pi / 4, math.pi / 8]),
      np.array([[[1.0, 0.0, 0.0], [0.0, np.cos(math.pi / 2), -np.sin(math.pi / 2)], [0.0, np.sin(math.pi / 2), np.cos(math.pi / 2)]],
                [[np.cos(math.pi / 4), 0.0, np.sin(math.pi / 4)], [0.0, 1.0, 0.0], [-np.sin(math.pi / 4), 0.0, np.cos(math.pi / 4)]],
                [[np.cos(math.pi / 8), -np.sin(math.pi / 8), 0.0], [np.sin(math.pi / 8), np.cos(math.pi / 8), 0.0], [0.0, 0.0, 1.0]]]))])
def test_rotation_batch_nxn(axis, alpha, expected):
    result = t.rotation(axis, alpha)
    np.testing.assert_allclose(result, expected, atol=1e-07, verbose=True)


@pytest.mark.parametrize('rotation, translation, expected', [
    (np.array([[1.0, 0.0, 0.0], [0.0, np.cos(math.pi / 2), -np.sin(math.pi / 2)], [0.0, np.sin(math.pi / 2), np.cos(math.pi / 2)]]), np.array([1.0, 2.0, 3.0]),
     np.array([[1.0, 0.0, 0.0, 1.0], [0.0, np.cos(math.pi / 2), -np.sin(math.pi / 2), 2.0], [0.0, np.sin(math.pi / 2),
                                                                                             np.cos(math.pi / 2), 3.0], [0.0, 0.0, 0.0, 1.0]])),
    (np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]), np.array([0, 0, 0]), np.array([[1, 2, 3, 0], [4, 5, 6, 0], [7, 8, 9, 0], [0, 0, 0, 1]])),
    (np.array([[11, 22, 33], [44, 55, 66], [77, 88, 99]]), np.array([6, 15, 24]), np.array([[11, 22, 33, 6], [44, 55, 66, 15], [77, 88, 99, 24], [0, 0, 0, 1]]))
])
def test_transformation_single(rotation, translation, expected):
    result = t.transformation(rotation, translation)
    np.testing.assert_array_equal(result, expected, verbose=True)


@pytest.mark.parametrize(
    'rotation, translation, expected',
    [(np.array([[[1, 2, 3], [4, 5, 6], [7, 8, 9]], [[11, 22, 33], [44, 55, 66], [77, 88, 99]]]), np.array([[0, 0, 0], [6, 15, 24]]),
      np.array([[[1, 2, 3, 0], [4, 5, 6, 0], [7, 8, 9, 0], [0, 0, 0, 1]], [[11, 22, 33, 6], [44, 55, 66, 15], [77, 88, 99, 24], [0, 0, 0, 1]]]))])
def test_transformation_batch(rotation, translation, expected):
    result = t.transformation(rotation, translation)
    np.testing.assert_array_equal(result, expected, verbose=True)