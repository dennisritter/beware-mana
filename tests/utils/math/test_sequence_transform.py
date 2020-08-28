"""Unittests for sequence transforms."""
import numpy as np
import pytest

import mana.utils.math.sequence_transform as st


@pytest.mark.parametrize('start_positions, expected', [(
    np.array([[[1, 2, 3], [1, 2, 3]], [[1, 2, 3], [1, 2, 3]]]),
    np.array([[[2, 1, 3], [2, 1, 3]], [[2, 1, 3], [2, 1, 3]]]),
)])
def test_swap_xy(start_positions, expected):
    """Test whether x and y axes are swapped."""
    swap_xy = st.SwapXY()
    transformed_positions = swap_xy(start_positions)
    np.testing.assert_equal(transformed_positions, expected)


@pytest.mark.parametrize('start_positions, expected', [(
    np.array([[[1, 2, 3], [1, 2, 3]], [[1, 2, 3], [1, 2, 3]]]),
    np.array([[[3, 2, 1], [3, 2, 1]], [[3, 2, 1], [3, 2, 1]]]),
)])
def test_swap_xz(start_positions, expected):
    """Test whether x and z axes are swapped."""
    swap_xz = st.SwapXZ()
    transformed_positions = swap_xz(start_positions)
    np.testing.assert_equal(transformed_positions, expected)


@pytest.mark.parametrize('start_positions, expected', [(
    np.array([[[1, 2, 3], [1, 2, 3]], [[1, 2, 3], [1, 2, 3]]]),
    np.array([[[1, 3, 2], [1, 3, 2]], [[1, 3, 2], [1, 3, 2]]]),
)])
def test_swap_yz(start_positions, expected):
    """Test whether y and z axes are swapped."""
    swap_yz = st.SwapYZ()
    transformed_positions = swap_yz(start_positions)
    np.testing.assert_equal(transformed_positions, expected)


@pytest.mark.parametrize('start_positions, expected', [(
    np.array([[[1, 2, 3], [1, 2, 3]], [[1, 2, 3], [1, 2, 3]]]),
    np.array([[[-1, 2, 3], [-1, 2, 3]], [[-1, 2, 3], [-1, 2, 3]]]),
)])
def test_flip_x(start_positions, expected):
    """Test whether x axis' values are flipped."""
    flip_x = st.FlipX()
    transformed_positions = flip_x(start_positions)
    np.testing.assert_equal(transformed_positions, expected)


@pytest.mark.parametrize('start_positions, expected', [(
    np.array([[[1, 2, 3], [1, 2, 3]], [[1, 2, 3], [1, 2, 3]]]),
    np.array([[[1, -2, 3], [1, -2, 3]], [[1, -2, 3], [1, -2, 3]]]),
)])
def test_flip_y(start_positions, expected):
    """Test whether y axis' values are flipped."""
    flip_y = st.FlipY()
    transformed_positions = flip_y(start_positions)
    np.testing.assert_equal(transformed_positions, expected)


@pytest.mark.parametrize('start_positions, expected', [(
    np.array([[[1, 2, 3], [1, 2, 3]], [[1, 2, 3], [1, 2, 3]]]),
    np.array([[[1, 2, -3], [1, 2, -3]], [[1, 2, -3], [1, 2, -3]]]),
)])
def test_flip_z(start_positions, expected):
    """Test whether z axis' values are flipped."""
    flip_z = st.FlipZ()
    transformed_positions = flip_z(start_positions)
    np.testing.assert_equal(transformed_positions, expected)
