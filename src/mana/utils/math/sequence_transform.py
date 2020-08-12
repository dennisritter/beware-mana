"""A collection of callable sequence transform classes."""

import numpy as np


class SwapXY:
    def __call__(self, positions: 'np.ndarray'):
        """Returns the given positions after swapping x-values with y-values
        
        Args:
            positions (np.ndarray): A time series of various 3-D positions 
            (ndim = 3) (shape = (n_frames, n_positions, 3))
        """
        return positions[:, :, [1, 0]]


class SwapXZ:
    def __call__(self, positions: 'np.ndarray'):
        """Returns the given positions after swapping x-values with z-values
        
        Args:
            positions (np.ndarray): A time series of various 3-D positions 
            (ndim = 3) (shape = (n_frames, n_positions, 3))
        """
        return positions[:, :, [2, 0]]


class SwapYZ:
    def __call__(self, positions: 'np.ndarray'):
        """Returns the given positions after swapping y-values with z-values
        
        Args:
            positions (np.ndarray): A time series of various 3-D positions 
            (ndim = 3) (shape = (n_frames, n_positions, 3))
        """
        return positions[:, :, [2, 1]]


class FlipX:
    def __call__(self, positions: 'np.ndarray'):
        """Returns the given positions after flipping the x-axis 
        (negating all x-values).
        
        Args:
            positions (np.ndarray): A time series of various 3-D positions 
            (ndim = 3) (shape = (n_frames, n_positions, 3))
        """
        return -positions[:, :, 0]


class FlipY:
    def __call__(self, positions: 'np.ndarray'):
        """Returns the given positions after flipping the x-axis 
        (negating all y-values).
        
        Args:
            positions (np.ndarray): A time series of various 3-D positions 
            (ndim = 3) (shape = (n_frames, n_positions, 3))
        """
        return -positions[:, :, 1]


class FlipZ:
    def __call__(self, positions: 'np.ndarray'):
        """Returns the given positions after flipping the x-axis 
        (negating all y-values).
        
        Args:
            positions (np.ndarray): A time series of various 3-D positions 
            (ndim = 3) (shape = (n_frames, n_positions, 3))
        """
        return -positions[:, :, 2]


## Just a template for a transform with basic memory
# class some_transform_that_needs_memory:
#     def __init__(self, some_memorized_param):
#         self.some_memorized_param = some_memorized_param

#     def __call__(positions: 'np.ndarray'):
#         pass  # returns some transformed positions

# TODO: Add sequence normalization transformations.