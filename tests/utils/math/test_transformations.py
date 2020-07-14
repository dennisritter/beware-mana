import pytest
import mana.utils.math.transformations as t
import numpy as np


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
