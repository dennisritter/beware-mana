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


class mka_to_iisy_bodyparts:
    """Callable transform class to flip z-values."""
    def __call__(self, positions: np.ndarray) -> np.ndarray:
        """Transforms MKA recorded motion data into the iisy body part format.

        Will Transform MKA human motion model (32 body parts) into the iisy
        body part model (16 body parts).

        # * MKA body part model (ORIGIN)
        # body_parts_mka = {
        #     "Pelvis": 0,
        #     "SpineNavel": 1,
        #     "SpineChest ": 2,
        #     "Neck": 3,
        #     "ClavicleLeft": 4,
        #     "ShoulderLeft": 5,
        #     "ElbowLeft ": 6,
        #     "WristLeft ": 7,
        #     "HandLeft ": 8,
        #     "HandTipLeft": 9,
        #     "ThumbLeft": 10,
        #     "ClavicleRight": 11,
        #     "ShoulderRight": 12,
        #     "ElbowRight": 13,
        #     "WristRight": 14,
        #     "HandRight": 15,
        #     "HandTipRight": 16,
        #     "ThumbRight": 17,
        #     "HipLeft": 18,
        #     "KneeLeft": 19,
        #     "AnkleLeft": 20,
        #     "FootLeft": 21,
        #     "HipRight": 22,
        #     "KneeRight": 23,
        #     "AnkleRight": 24,
        #     "FootRight": 25,
        #     "Head": 26,
        #     "Nose": 27,
        #     "EyeLeft": 28,
        #     "EarLeft": 29,
        #     "EyeRight": 30,
        #     "EarRight": 31
        # }

        # * IISY body part model (TARGET)
        # body_parts_iisy = {
        #     "head": 0,
        #     "neck": 1,
        #     "shoulder_l": 2,
        #     "shoulder_r": 3,
        #     "elbow_l": 4,
        #     "elbow_r": 5,
        #     "wrist_l": 6,
        #     "wrist_r": 7,
        #     "torso": 8,
        #     "pelvis": 9,
        #     "hip_l": 10,
        #     "hip_r": 11,
        #     "knee_l": 12,
        #     "knee_r": 13,
        #     "ankle_l": 14,
        #     "ankle_r": 15,
        # }

        Args:
            positions (np.ndarray): A time series of various 3-D positions
            (ndim = 3) (shape = (n_frames, n_positions, 3))

        Returns:
            np.ndarray: The transformed positions array.
        """
        transformed_positions = np.zeros((len(positions), 16, 3))
        transformed_positions[:, 0, :] = positions[:, 26, :]
        transformed_positions[:, 1, :] = positions[:, 3, :]
        transformed_positions[:, 2, :] = positions[:, 5, :]
        transformed_positions[:, 3, :] = positions[:, 12, :]
        transformed_positions[:, 4, :] = positions[:, 6, :]
        transformed_positions[:, 5, :] = positions[:, 13, :]
        transformed_positions[:, 6, :] = positions[:, 8, :]
        transformed_positions[:, 7, :] = positions[:, 14, :]
        transformed_positions[:, 8, :] = positions[:, 1, :]
        transformed_positions[:, 9, :] = positions[:, 0, :]
        transformed_positions[:, 10, :] = positions[:, 18, :]
        transformed_positions[:, 11, :] = positions[:, 22, :]
        transformed_positions[:, 12, :] = positions[:, 19, :]
        transformed_positions[:, 13, :] = positions[:, 23, :]
        transformed_positions[:, 14, :] = positions[:, 20, :]
        transformed_positions[:, 15, :] = positions[:, 24, :]

        return transformed_positions


## Just a template for a transform with basic memory
# class some_transform_that_needs_memory:
#     def __init__(self, some_memorized_param):
#         self.some_memorized_param = some_memorized_param

#     def __call__(positions: np.ndarray) -> np.ndarray:
#         pass  # returns some transformed positions
