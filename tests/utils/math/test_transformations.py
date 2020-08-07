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


@pytest.mark.parametrize('rotation, translation, expected',
                         [(None, np.array([1, 2, 3]), np.array([[1.0, 0.0, 0.0, 1.0], [0.0, 1.0, 0.0, 2.0], [0.0, 0.0, 1.0, 3.0], [0, 0, 0, 1]]))])
def test_transformation_no_rotation(rotation, translation, expected):
    result = t.transformation(rotation, translation)
    np.testing.assert_array_equal(result, expected, verbose=True)


@pytest.mark.parametrize(
    'rotation, translation, expected',
    [(np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]), None, np.array([[1.0, 2.0, 3.0, 0.0], [4.0, 5.0, 6.0, 0.0], [7.0, 8.0, 9.0, 0.0], [0, 0, 0, 1]]))])
def test_transformation_no_translation(rotation, translation, expected):
    result = t.transformation(rotation, translation)
    np.testing.assert_array_equal(result, expected, verbose=True)


@pytest.mark.parametrize(
    'rotation, translation, expected',
    [(np.array([[[1, 2, 3], [4, 5, 6], [7, 8, 9]], [[11, 22, 33], [44, 55, 66], [77, 88, 99]]]), np.array([[0, 0, 0], [6, 15, 24]]),
      np.array([[[1, 2, 3, 0], [4, 5, 6, 0], [7, 8, 9, 0], [0, 0, 0, 1]], [[11, 22, 33, 6], [44, 55, 66, 15], [77, 88, 99, 24], [0, 0, 0, 1]]]))])
def test_transformation_batch(rotation, translation, expected):
    result = t.transformation(rotation, translation)
    np.testing.assert_array_equal(result, expected, verbose=True)


@pytest.mark.parametrize('v1, v2, expected',
                         [(np.array([1, 2, 3]), np.array([1, 1, 1]), np.array([-1.0, 2.0, -1.0]) / np.linalg.norm(np.array([-1.0, 2.0, -1.0]))),
                          (np.array([1, 1, 1]), np.array([1, 2, 3]), np.array([1.0, -2.0, 1.0]) / np.linalg.norm(np.array([1.0, -2.0, 1.0])))])
def test_orthogonal_vector_single(v1, v2, expected):
    result = t.orthogonal_vector(v1, v2)
    np.testing.assert_allclose(result, expected, atol=1e-07, verbose=True)


# For parallel vectors, don't assert the actual vector but whether the dot product between v1/v2 and the returned vector is 0 (90 degreees)
@pytest.mark.parametrize('v1, v2, expected', [(np.array([1, 1, 1]), np.array([1, 1, 1]), 0), (np.array([1, 2, 3]), np.array([1, 2, 3]), 0)])
def test_orthogonal_vector_single_parallel(v1, v2, expected):
    result_vec = t.orthogonal_vector(v1, v2)
    assert t.dot(result_vec, v1) == pytest.approx(0.0)
    assert t.dot(result_vec, v2) == pytest.approx(0.0)


@pytest.mark.parametrize(
    'v1, v2, expected',
    [(np.array([[1, 1, 1], [1, 2, 3]]), np.array([[1, 2, 3], [1, 1, 1]]),
      np.array([[1.0, -2.0, 1.0] / np.linalg.norm(np.array([1.0, -2.0, 1.0])), [-1.0, 2.0, -1.0] / np.linalg.norm(np.array([-1.0, 2.0, -1.0]))]))])
def test_orthogonal_vector_multi(v1, v2, expected):
    result = t.orthogonal_vector(v1, v2)
    np.testing.assert_allclose(result, expected, atol=1e-07, verbose=True)


# For parallel vectors, don't assert the actual vector but whether the dot product between v1/v2 and the returned vector is 0 (90 degreees)
@pytest.mark.parametrize('v1, v2, expected', [(np.array([[1, 1, 1], [1, 2, 3]]), np.array([[1, 1, 1], [1, 2, 3]]), np.array([0.0, 0.0]))])
def test_orthogonal_vector_multi_parallel(v1, v2, expected):
    result = t.orthogonal_vector(v1, v2)
    np.testing.assert_allclose(t.dot(result, v1), expected, atol=1e-07, verbose=True)
    np.testing.assert_allclose(t.dot(result, v2), expected, atol=1e-07, verbose=True)


# Expected result vector is v_from multiplied with the returned rotation, which must be v_to
@pytest.mark.parametrize('v_from, v_to, expected', [(np.array([1, 1, 1]), np.array([1, 2, 3]), np.array([1, 2, 3])),
                                                    (np.array([1, 0, 0]), np.array([0, 1, 0]), np.array([0, 1, 0]))])
def test_rotation_from_vectors_single(v_from, v_to, expected):
    result = t.rotation_from_vectors(v_from, v_to)
    np.testing.assert_allclose(t.norm(v_from) @ result, t.norm(expected), atol=1e-7, verbose=True)


@pytest.mark.parametrize('v_from, v_to, expected', [(np.array([1, 1, 1]), np.array([1, 1, 1]), np.identity(3)),
                                                    (np.array([1, 2, 3]), np.array([1, 2, 3]), np.identity(3))])
def test_rotation_from_vectors_single_equals(v_from, v_to, expected):
    result = t.rotation_from_vectors(v_from, v_to)
    np.testing.assert_array_equal(result, expected)


@pytest.mark.parametrize(
    'v_from, v_to, expected',
    [(np.array([[1, 1, 1], [1, 2, 3], [0.1, 3.3, 7.0]]), np.array([[1, 1, 1], [1, 2, 3], [0.1, 3.3, 7.0]]), np.array([[1, 1, 1], [1, 2, 3], [0.1, 3.3, 7.0]]))])
def test_rotation_from_vectors_multi(v_from, v_to, expected):
    result = t.rotation_from_vectors(v_from, v_to)
    np.testing.assert_allclose(np.array([t.norm(v_from[idx]) @ result[idx] for idx, _ in enumerate(result)]), t.norm(expected), atol=1e-7, verbose=True)


@pytest.mark.parametrize('v_from, v_to, expected',
                         [(np.array([[1, 1, 1], [1, 2, 3]]), np.array([[1, 1, 1], [1, 2, 3]]), np.array([np.identity(3), np.identity(3)]))])
def test_rotation_from_vectors_multi_equals(v_from, v_to, expected):
    result = t.rotation_from_vectors(v_from, v_to)
    np.testing.assert_array_equal(result, expected)


@pytest.mark.parametrize('m1, m2, expected',
                         [(np.array([[[1, 2, 3], [4, 5, 6], [7, 8, 9]], [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
                                     ]), np.array([[[1, 0, 0], [0, 1, 0], [0, 0, 1]], [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
                                                   ]), np.array([[[1, 2, 3], [4, 5, 6], [7, 8, 9]], [[30, 36, 42], [66, 81, 96], [102, 126, 150]]]))])
def test_bmm_nxn_3x3(m1, m2, expected):
    result = t.bmm_nxn(m1, m2)
    np.testing.assert_array_equal(result, expected)


@pytest.mark.parametrize(
    'm1, m2, expected',
    [(np.array([[[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]], [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]]
                ]), np.array([[[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]]),
      np.array([[[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]], [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]]]))])
def test_bmm_nxn_4x4(m1, m2, expected):
    result = t.bmm_nxn(m1, m2)
    np.testing.assert_array_equal(result, expected)


@pytest.mark.parametrize('m, v, expected', [(np.array([[[1, 2, 3], [4, 5, 6], [7, 8, 9]], [[1, 2, 3], [4, 5, 6], [7, 8, 9]]]), np.array(
    [[1, 1, 1], [1, 2, 3]]), np.array([[6, 15, 24], [14, 32, 50]]))])
def test_bmvm_3x3(m, v, expected):
    result = t.bmvm(m, v)
    np.testing.assert_array_equal(result, expected)


@pytest.mark.parametrize('m, v, expected',
                         [(np.array([[[1, 2, 3, 0], [4, 5, 6, 0], [7, 8, 9, 0], [0, 0, 0, 1]], [[1, 2, 3, 0], [4, 5, 6, 0], [7, 8, 9, 0], [0, 0, 0, 1]]
                                     ]), np.array([[1, 1, 1, 1], [1, 2, 3, 1]]), np.array([[6, 15, 24, 1], [14, 32, 50, 1]]))])
def test_bmvm_4x4(m, v, expected):
    result = t.bmvm(m, v)
    np.testing.assert_array_equal(result, expected)
