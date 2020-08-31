"""A collection of callable sequence transform classes."""
import numpy as np


class SwapXY:
    """Callable transform class to swap x-values with y-values."""
    def __call__(self, positions: np.ndarray) -> np.ndarray:
        """Returns the given positions after swapping x-values with y-values

        Args:
            positions (np.ndarray): A time series of various 3-D positions
            (ndim = 3) (shape = (n_frames, n_positions, 3))

        Returns:
            np.ndarray: The transformed positions array.
        """
        positions[:, :, [0, 1]] = positions[:, :, [1, 0]]
        return positions


class SwapXZ:
    """Callable transform class to swap x-values with z-values."""
    def __call__(self, positions: np.ndarray) -> np.ndarray:
        """Returns the given positions after swapping x-values with z-values

        Args:
            positions (np.ndarray): A time series of various 3-D positions
            (ndim = 3) (shape = (n_frames, n_positions, 3))

        Returns:
            np.ndarray: The transformed positions array.
        """
        positions[:, :, [0, 2]] = positions[:, :, [2, 0]]
        return positions


class SwapYZ:
    """Callable transform class to swap y-values with z-values."""
    def __call__(self, positions: np.ndarray) -> np.ndarray:
        """Returns the given positions after swapping y-values with z-values

        Args:
            positions (np.ndarray): A time series of various 3-D positions
            (ndim = 3) (shape = (n_frames, n_positions, 3))

        Returns:
            np.ndarray: The transformed positions array.
        """
        positions[:, :, [1, 2]] = positions[:, :, [2, 1]]
        return positions


class FlipX:
    """Callable transform class to flip x-values."""
    def __call__(self, positions: np.ndarray) -> np.ndarray:
        """Returns the given positions after flipping the x-axis
        (negating all x-values).

        Args:
            positions (np.ndarray): A time series of various 3-D positions
            (ndim = 3) (shape = (n_frames, n_positions, 3))

        Returns:
            np.ndarray: The transformed positions array.
        """
        positions[:, :, 0] = -positions[:, :, 0]
        return positions


class FlipY:
    """Callable transform class to flip y-values."""
    def __call__(self, positions: np.ndarray) -> np.ndarray:
        """Returns the given positions after flipping the x-axis
        (negating all y-values).

        Args:
            positions (np.ndarray): A time series of various 3-D positions
            (ndim = 3) (shape = (n_frames, n_positions, 3))

        Returns:
            np.ndarray: The transformed positions array.
        """
        positions[:, :, 1] = -positions[:, :, 1]
        return positions


class FlipZ:
    """Callable transform class to flip z-values."""
    def __call__(self, positions: np.ndarray) -> np.ndarray:
        """Returns the given positions after flipping the x-axis
        (negating all y-values).

        Args:
            positions (np.ndarray): A time series of various 3-D positions
            (ndim = 3) (shape = (n_frames, n_positions, 3))

        Returns:
            np.ndarray: The transformed positions array.
        """
        positions[:, :, 2] = -positions[:, :, 2]
        return positions


## Just a template for a transform with basic memory
# class some_transform_that_needs_memory:
#     def __init__(self, some_memorized_param):
#         self.some_memorized_param = some_memorized_param

#     def __call__(positions: np.ndarray) -> np.ndarray:
#         pass  # returns some transformed positions
