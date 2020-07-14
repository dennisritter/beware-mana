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
@pytest.mark.parametrize('v1, v2, expected', [(np.array([1, 1, 1]), np.array([1, 1, 1]), 0.0), (np.array([1, 0, 0]), np.array([0, 1, 0]), 90.0),
                                              (np.array([1, 0, 0]), np.array([1, 1, 0]), 45.0)])
def test_get_angle_single(v1, v2, expected):
    t.get_angle(v1, v2) == expected


@pytest.mark.parametrize('v1, v2, expected',
                         [(np.array([[1, 1, 1], [1, 0, 0], [1, 0, 0]]), np.array([[1, 1, 1], [0, 1, 0], [1, 1, 0]]), np.array([0.0, 90.0, 45.0]))])
def test_get_angle_batch(v1, v2, expected):
    np.array_equal(t.get_angle(v1, v2), expected)


# dot
@pytest.mark.parametrize('v1, v2, expected', [(np.array([1, 1, 1]), np.array([1, 1, 1]), 0.0), (np.array([1, 0, 0]), np.array([0, 1, 0]), math.pi / 2),
                                              (np.array([1, 0, 0]), np.array([1, 1, 0]), math.pi / 4)])
def test_dot_single(v1, v2, expected):
    t.dot(v1, v2) == expected


@pytest.mark.parametrize('v1, v2, expected',
                         [(np.array([[1, 1, 1], [1, 0, 0], [1, 0, 0]]), np.array([[1, 1, 1], [0, 1, 0], [1, 1, 0]]), np.array([0.0, math.pi / 2, math.pi / 4]))]
                         )
def test_dot_batch(v1, v2, expected):
    np.array_equal(t.dot(v1, v2), expected)
