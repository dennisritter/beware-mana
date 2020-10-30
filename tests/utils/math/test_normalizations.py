"""Contains the test code for the mana.utils.math.normalization module."""

import pytest
import numpy as np
import mana.utils.math.normalizations as n


@pytest.mark.parametrize(
    'array, mean, expected',
    [
        # Means = [2.5, 3.5, 4.5]
        (np.array([
            [[1, 2, 3], [4, 5, 6]],
            [[1, 2, 3], [4, 5, 6]],
            [[1, 2, 3], [4, 5, 6]],
        ]), None,
         np.array([
             [[-1.5, -1.5, -1.5], [1.5, 1.5, 1.5]],
             [[-1.5, -1.5, -1.5], [1.5, 1.5, 1.5]],
             [[-1.5, -1.5, -1.5], [1.5, 1.5, 1.5]],
         ])),
        # Means = [3, 2, 4]
        (np.array([
            [[2, 3, 4], [5, 1, 2], [3, 3, 3], [2, 1, 7]],
            [[1, 2, 3], [3, 2, 1], [4, 0, 3], [4, 4, 9]],
            [[3, 5, 4], [4, 0, 1], [2, 1, 2], [3, 2, 9]],
        ]), None,
         np.array([
             [[-1, 1, 0], [2, -1, -2], [0, 1, -1], [-1, -1, 3]],
             [[-2, 0, -1], [0, 0, -3], [1, -2, -1], [1, 2, 5]],
             [[0, 3, 0], [1, -2, -3], [-1, -1, -2], [0, 0, 5]],
         ])),
        # Means = [1.5, 0.5, 1.0, 3.0]
        (np.array([
            [[1, 1, 1, 3], [2, 0, 1, 3]],
            [[3, 1, 2, 5], [0, 0, 0, 1]],
            [[0, 0, 0, 4], [3, 1, 2, 2]],
            [[-1, 3, 0, 5], [4, -2, 2, 1]],
            [[3, -1, 1, 5], [0, 2, 1, 1]],
        ]), None,
         np.array([
             [[-0.5, 0.5, 0.0, 0.0], [0.5, -0.5, 0.0, 0.0]],
             [[1.5, 0.5, 1.0, 2.0], [-1.5, -0.5, -1.0, -2.0]],
             [[-1.5, -0.5, -1.0, 1.0], [1.5, 0.5, 1.0, -1.0]],
             [[-2.5, 2.5, -1.0, 2.0], [2.5, -2.5, 1.0, -2.0]],
             [[1.5, -1.5, 0.0, 2.0], [-1.5, 1.5, 0.0, -2.0]],
         ])),
        (np.array([
            [[1, 1, 1, 0], [2, 0, 3, 0]],
            [[3, 1, 2, 3], [0, 1, 0, 1]],
            [[0, 0, 0, 0], [5, 1, 4, 2]],
            [[3, 3, 1, 0], [4, 3, 2, 1]],
            [[3, 1, 0, 5], [1, 1, 0, 0]],
        ]),
         np.array([
             [[1, 1, 1, 0], [2, 0, 3, 0]],
             [[3, 1, 2, 3], [0, 1, 0, 1]],
             [[0, 0, 0, 0], [5, 1, 4, 2]],
             [[3, 3, 1, 0], [4, 3, 2, 1]],
             [[3, 1, 0, 5], [1, 1, 0, 0]],
         ]),
         np.array([
             [[-1.2, -0.2, -0.3, -1.2], [-0.2, -1.2, 1.7, -1.2]],
             [[0.8, -0.2, 0.7, 1.8], [-2.2, -0.2, -1.3, -0.2]],
             [[-2.2, -1.2, -1.3, -1.2], [2.8, -0.2, 2.7, 0.8]],
             [[0.8, 1.8, -0.3, -1.2], [1.8, 1.8, 0.7, -0.2]],
             [[0.8, -0.2, -1.3, 3.8], [-1.2, -0.2, -1.3, -1.2]],
         ])),
        (np.array([
            [[1, 1, 1, 0], [2, 0, 3, 0], [3, 1, 2, 3], [0, 1, 0, 1],
             [0, 0, 0, 0]],
            [[5, 1, 4, 2], [3, 3, 1, 0], [4, 3, 2, 1], [3, 1, 0, 5],
             [1, 1, 0, 0]],
        ]),
         np.array([
             [[1, 1, 1, 0], [2, 0, 3, 0], [3, 1, 2, 3], [0, 1, 0, 1],
              [0, 0, 0, 0]],
             [[5, 1, 4, 2], [3, 3, 1, 0], [4, 3, 2, 1], [3, 1, 0, 5],
              [1, 1, 0, 0]],
         ]),
         np.array([
             [[-1.2, -0.2, -0.3, -1.2], [-0.2, -1.2, 1.7, -1.2],
              [0.8, -0.2, 0.7, 1.8], [-2.2, -0.2, -1.3, -0.2],
              [-2.2, -1.2, -1.3, -1.2]],
             [[2.8, -0.2, 2.7, 0.8], [0.8, 1.8, -0.3, -1.2],
              [1.8, 1.8, 0.7, -0.2], [0.8, -0.2, -1.3, 3.8],
              [-1.2, -0.2, -1.3, -1.2]],
         ])),
    ],
)
def test_pose_mean(array, mean, expected):
    """Tests if the (combined) mean for all frames is computed and subtracted
    accordingly."""
    # using pytest approx to avoid floating point arithmetic issues
    assert pytest.approx(n.pose_mean(array, mean)) == expected
    # np.testing.assert_array_almost_equal(n.pose_mean2(array, mean) == expected)


@pytest.mark.parametrize(
    'array, pos, expected',
    [
        (np.array([
            [[2, 3, 4], [5, 1, 2], [3, 3, 3], [1, 0, 1]],
            [[1, 2, 3], [3, 2, 1], [4, 0, 3], [5, 4, 2]],
            [[3, 5, 4], [0, 0, 1], [2, 1, 2], [3, 2, 1]],
        ]), np.array([5, 1, 2]),
         np.array([
             [[-3, 2, 2], [0, 0, 0], [-2, 2, 1], [-4, -1, -1]],
             [[-4, 1, 1], [-2, 1, -1], [-1, -1, 1], [0, 3, 0]],
             [[-2, 4, 2], [-5, -1, -1], [-3, 0, 0], [-2, 1, -1]],
         ])),
        (np.array([
            [[2, 3, 4], [5, 1, 2], [3, 3, 3], [1, 0, 1]],
            [[1, 2, 3], [3, 2, 1], [4, 0, 3], [5, 4, 2]],
            [[3, 5, 4], [0, 0, 1], [2, 1, 2], [3, 2, 1]],
        ]), np.array([
            [5, 1, 2],
            [3, 2, 1],
            [0, 0, 1],
        ]),
         np.array([
             [[-3, 2, 2], [0, 0, 0], [-2, 2, 1], [-4, -1, -1]],
             [[-2, 0, 2], [0, 0, 0], [1, -2, 2], [2, 2, 1]],
             [[3, 5, 3], [0, 0, 0], [2, 1, 1], [3, 2, 0]],
         ])),
        (np.array([
            [[1, 1, 1, 0], [2, 0, 3, 0], [3, 1, 2, 3], [0, 1, 0, 1],
             [0, 0, 0, 0]],
            [[5, 1, 4, 2], [3, 3, 1, 0], [4, 3, 2, 1], [3, 1, 0, 5],
             [1, 1, 0, 0]],
        ]), np.array([3, 1, 2, 3]),
         np.array([
             [[-2, 0, -1, -3], [-1, -1, 1, -3], [0, 0, 0, 0], [-3, 0, -2, -2],
              [-3, -1, -2, -3]],
             [[2, 0, 2, -1], [0, 2, -1, -3], [1, 2, 0, -2], [0, 0, -2, 2],
              [-2, 0, -2, -3]],
         ])),
        (np.array([
            [[1, 1, 1, 0], [2, 0, 3, 0], [3, 1, 2, 3], [0, 1, 0, 1],
             [0, 0, 0, 0]],
            [[5, 1, 4, 2], [3, 3, 1, 0], [4, 3, 2, 1], [3, 1, 0, 5],
             [1, 1, 0, 0]],
        ]), np.array([
            [3, 1, 2, 3],
            [4, 3, 2, 1],
        ]),
         np.array([
             [[-2, 0, -1, -3], [-1, -1, 1, -3], [0, 0, 0, 0], [-3, 0, -2, -2],
              [-3, -1, -2, -3]],
             [[1, -2, 2, 1], [-1, 0, -1, -1], [0, 0, 0, 0], [-1, -2, -2, 4],
              [-3, -2, -2, -1]],
         ])),
    ],
)
def test_pose_position(array, pos, expected):
    """Tests if the positions for all frames will be translated by the given
    vector or array of vectors."""
    assert (n.pose_position(array, pos) == expected).all()


@pytest.mark.parametrize(
    'array, v_from, v_to, axis, origin, expected',
    [
        (
            np.array([[[3, 2, 2], [3, 1, 3]]]),
            np.array([0, 1, -1]),  # difference from v1 to v2
            np.array([1, 0, 0]),  # rotate towards x axis
            np.array([0, 0, 1]),  # rotate around z axis
            np.array([0, 0, 0]),  # origin
            np.array([[[-2, 3, 2], [-1, 3, 3]]]),  # simple 90 degree rotation
        ),  # simple rotation
        (
            np.array([[[3, 2, 2], [3, 1, 3]]]),
            np.array([0, 1, -1]),
            np.array([1, 0, 0]),
            np.array([0, 0, 1]),
            np.array([3, 2, 2]),  # origin equals v1
            np.array([[[3, 2, 2], [4, 2, 3]]]),  # 90 degree rotation around v1
        ),  # inplace rotation -> different origin
        (
            np.array([[[4, 1, 2], [2, 3, 2]]]),
            np.array([2, -2, 0]),
            np.array([1, 0, 0]),
            np.array([0, 1, 0]),  # rotate around y
            np.array([0, 0, 0]),
            np.array([[[4, 1, 2], [2, 3, 2]]]),  # v1 and v2 are parallel to y
        ),  # different rotation axis
        (
            np.array([[[3, 2, 2], [3, 1, 3]], [[3, 2, 2], [3, 1, 3]]]),
            np.array([0, 1, -1]),
            np.array([1, 0, 0]),
            np.array([0, 0, 1]),
            np.array([0, 0, 0]),
            np.array([[[-2, 3, 2], [-1, 3, 3]], [[-2, 3, 2], [-1, 3, 3]]]),
        ),  # 2 frames (1 v_from, 1 v_to) -> apply first rotation to all
        (
            np.array([[[3, 2, 2], [3, 1, 3]], [[3, 2, 2], [3, 1, 3]]]),
            np.array([[0, 1, -1], [0, 1, -1]]),
            np.array([[1, 0, 0], [1, 0, 0]]),
            np.array([0, 0, 1]),
            np.array([0, 0, 0]),
            np.array([[[-2, 3, 2], [-1, 3, 3]], [[-2, 3, 2], [-1, 3, 3]]]),
        ),  # 2 frames (2 v_from, 2 v_to) -> compute and apply rotation to each
        (
            np.array([[[3, 2, 2], [3, 1, 3]], [[3, 2, 2], [3, 1, 3]]]),
            np.array([[0, 1, -1], [0, 1, -1]]),
            np.array([1, 0, 0]),
            np.array([0, 0, 1]),
            np.array([0, 0, 0]),
            np.array([[[-2, 3, 2], [-1, 3, 3]], [[-2, 3, 2], [-1, 3, 3]]]),
        ),  # 2 frames (2 v_from, 1 v_to) -> broadcast 
        (
            np.array([[[3, 2, 2], [3, 1, 3]], [[3, 2, 2], [3, 1, 3]]]),
            np.array([0, 1, -1]),
            np.array([[1, 0, 0], [1, 0, 0]]),
            np.array([0, 0, 1]),
            np.array([0, 0, 0]),
            np.array([[[-2, 3, 2], [-1, 3, 3]], [[-2, 3, 2], [-1, 3, 3]]]),
        ),  # 2 frames (1 v_from, 2 v_to) -> broadcast 
        (
            np.array([[[3, 2, 2], [3, 1, 3]], [[4, 1, 2], [2, 3, 2]]]),
            np.array([[0, 1, -1], [2, -2, 0]]),
            np.array([1, 0, 0]),
            np.array([[0, 0, 1], [0, 1, 0]]),
            np.array([0, 0, 0]),
            np.array([[[-2, 3, 2], [-1, 3, 3]], [[4, 1, 2], [2, 3, 2]]]),
        ),  # 2 frames 2 axis
        (
            np.array([[[3, 2, 2], [3, 1, 3]], [[3, 2, 2], [3, 1, 3]]]),
            np.array([0, 1, -1]),
            np.array([1, 0, 0]),
            np.array([0, 0, 1]),
            np.array([[0, 0, 0], [3, 2, 2]]),
            np.array([[[-2, 3, 2], [-1, 3, 3]], [[3, 2, 2], [4, 2, 3]]]),
        ),  # 2 frames 2 origin
    ])
def test_pose_orientation(array, v_from, v_to, axis, origin, expected):
    """Tests if the positions will be rotated toward the plane of the given
    plane normal."""
    assert pytest.approx(n.pose_orientation(array, v_from, v_to, axis,
                                            origin)) == expected
