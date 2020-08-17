"""Contains the test code for the mana.utils.math.normalization module."""

import pytest
import numpy as np
import mana.utils.math.normalizations as n


@pytest.mark.parametrize(
    'var, mean, expected',
    [
        (np.array([
            [[2, 3, 4], [5, 1, 2], [3, 3, 3], [1, 0, 1]],
            [[1, 2, 3], [3, 2, 1], [4, 0, 3], [5, 4, 2]],
            [[3, 5, 4], [0, 0, 1], [2, 1, 2], [3, 2, 1]],
        ]), None,
         np.array([
             [[-0.75, 1.25, 1.5], [2.25, -0.75, -0.5], [0.25, 1.25, 0.5],
              [-1.75, -1.75, -1.5]],
             [[-1.75, 0.25, 0.5], [0.25, 0.25, -1.5], [1.25, -1.75, 0.5],
              [2.25, 2.25, -0.5]],
             [[0.25, 3.25, 1.5], [-2.75, -1.75, -1.5], [-0.75, -0.75, -0.5],
              [0.25, 0.25, -1.5]],
         ])),
        (np.array([
            [[1, 1, 1, 0], [2, 0, 3, 0]],
            [[3, 1, 2, 3], [0, 1, 0, 1]],
            [[0, 0, 0, 0], [5, 1, 4, 2]],
            [[3, 3, 1, 0], [4, 3, 2, 1]],
            [[3, 1, 0, 5], [1, 1, 0, 0]],
        ]), None,
         np.array([
             [[-0.5, 0.5, -1, 0], [0.5, -0.5, 1, 0]],
             [[1.5, 0.5, 0, 3], [-1.5, 0.5, -2, 1]],
             [[-1.5, -0.5, -2, 0], [3.5, 0.5, 2, 2]],
             [[1.5, 2.5, -1, 0], [2.5, 2.5, 0, 1]],
             [[1.5, 0.5, -2, 5], [-0.5, 0.5, -2, 0]],
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
def test_pose_mean(var, mean, expected):
    """Tests if the (combined) mean for all frames is computed and subtracted
    accordingly."""
    # using pytest approx to avoid floating point arithmetic issues
    assert pytest.approx(n.pose_mean(var, mean) == expected)


@pytest.mark.parametrize(
    'var, pos, expected',
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
def test_pose_position(var, pos, expected):
    """Tests if the positions for all frames will be translated by the given
    vector or array of vectors."""
    assert (n.pose_position(var, pos) == expected).all()


@pytest.mark.parametrize(
    'var, rotation_vectors, orthogonal_vectors, origin_vector, expected',
    [
        (
            np.array([[[2, 3, 2], [4, 1, 2], [2, 2, 3]]]),
            np.array([-2, 2, 0]),
            np.array([0, 1, 0]),
            np.array([0, 0, 0]),
            np.array([
                [-0.07071, 3.5355, 2],
                [2.1213, 3.5355, 2],
                [0, 2.8284, 3],
            ]),
        ),  # origin rotation
        (
            np.array([[[2, 3, 2], [4, 1, 2], [2, 2, 3]]]),
            np.array([-2, 2, 0]),
            np.array([0, 1, 0]),
            np.array([2, 3, 2]),
            np.array([[2, 3, 2], [4.8284, 3, 2], [2.7071, 2.8929, 3]]),
        ),  # inplace rotation -> different origin
        (
            np.array([[[2, 3, 2], [4, 1, 2], [2, 2, 3]]]),
            np.array([-2, 2, 0]),
            np.array([0, 0, 1]),
            np.array([0, 0, 0]),
            np.array([[2, 3, 2], [4, 1, 2], [2, 2, 3]]),
        ),  # xy plane -> different normal vector
        (
            np.array([[[2, 3, 2], [4, 1, 2], [2, 2, 3]],
                      [[3, 2, 2], [3, 1, 3], [1, 3, 2]]]),
            np.array([-2, 2, 0]),
            np.array([0, 1, 0]),
            np.array([0, 0, 0]),
            np.array([[
                [-0.07071, 3.5355, 2],
                [2.1213, 3.5355, 2],
                [0, 2.8284, 3],
            ], [
                [0.7071, 3.5355, 2],
                [1.4142, 2.8284, 3],
                [-1.4142, 2.8284, 2],
            ]]),
        ),  # 2 frames -> apply first rotation to all
        (
            np.array([[[2, 3, 2], [4, 1, 2], [2, 2, 3]],
                      [[3, 2, 2], [3, 1, 3], [1, 3, 2]]]),
            np.array([[-2, 2, 0], [0, 1, -1]]),
            np.array([0, 1, 0]),
            np.array([0, 0, 0]),
            np.array([[
                [-0.07071, 3.5355, 2],
                [2.1213, 3.5355, 2],
                [0, 2.8284, 3],
            ], [
                [0.7071, 3.5355, 2],
                [1.4142, 2.8284, 3],
                [-1.4142, 2.8284, 2],
            ]]),
        ),  # 2 frames 2 rotation -> compute and apply rotation to each
        (
            np.array([[[2, 3, 2], [4, 1, 2], [2, 2, 3]],
                      [[3, 2, 2], [3, 1, 3], [1, 3, 2]]]),
            np.array([[-2, 2, 0], [0, 1, -1]]),
            np.array([[0, 1, 0], [0, 1, 0]]),
            np.array([0, 0, 0]),
            np.array([[
                [-0.07071, 3.5355, 2],
                [2.1213, 3.5355, 2],
                [0, 2.8284, 3],
            ], [
                [0.7071, 3.5355, 2],
                [1.4142, 2.8284, 3],
                [-1.4142, 2.8284, 2],
            ]]),
        ),  # 2 frames 2 orthogonal
        (
            np.array([[[2, 3, 2], [4, 1, 2], [2, 2, 3]],
                      [[3, 2, 2], [3, 1, 3], [1, 3, 2]]]),
            np.array([-2, 2, 0]),
            np.array([0, 1, 0]),
            np.array([[0, 0, 0], [0, 0, 0]]),
            np.array([[
                [-0.07071, 3.5355, 2],
                [2.1213, 3.5355, 2],
                [0, 2.8284, 3],
            ], [
                [0.7071, 3.5355, 2],
                [1.4142, 2.8284, 3],
                [-1.4142, 2.8284, 2],
            ]]),
        ),  # 2 frames 2 origin
    ])
def test_pose_orientation(var, rotation_vectors, orthogonal_vectors,
                          origin_vector, expected):
    """Tests if the positions will be rotated toward the plane of the given
    orthogonal vector."""
    assert pytest.approx(
        n.pose_orientation(var, rotation_vectors, orthogonal_vectors,
                           origin_vector) == expected)
