"""Contains callable sequence transform classes as well as a collection class
for such. The collection class 'SequenceTransforms' also contains methods for
common transformation chains."""
from typing import Callable, List

import numpy as np


class SequenceTransforms:
    """A collection class for SequenceTransforms."""
    def __init__(self, transforms: List[Callable[[np.ndarray], np.ndarray]]):
        """
        Args:
            List[Callable[[np.ndarray], np.ndarray]]: List of callable
                transforms which in- and outputs np.ndarrays.
        """
        self.transforms = transforms

    # * IISY body part model
    # body_parts = {
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

    @staticmethod
    def mir_to_iisy() -> List[Callable[[np.ndarray], np.ndarray]]:
        """Returns a list of transforms that transform positions from the MIR
        coordinate system to the IISY coordinate system.

        Note that the described coordinate system directions are described
        relative to the human actors viewing direction in the anatomical
        normal position
        (https://en.wikipedia.org/wiki/Standard_anatomical_position).
        Consequently, flipping and swapping axes may also flip the directions
        of other axes too. A solution to prevent this is:
        1. Flip all axes so that they point RIGHT, FRONT and BACK
        (or your targeted directions)
        2. Swap necessary axes

        MIR Coordinate System:
            X = Right
            Y = Up
            Z = Back

        IISY Coordinate System:
            X = Right
            Y = Front
            Z = Up
        formal transform: [0, 1, 2] -> [0, -2, 1]

        Returns:
            List[Callable[[np.ndarray], np.ndarray]]: List of callable
                transforms which in- and outputs np.ndarrays.
        """
        # * MIR body part model
        # body_parts_mir = {
        #     "LeftWrist": 0,
        #     "LeftElbow": 1,
        #     "LeftShoulder": 2,
        #     "Neck": 3,
        #     "Torso": 4,
        #     "Waist": 5,
        #     "LeftAnkle": 6,
        #     "LeftKnee": 7,
        #     "LeftHip": 8,
        #     "RightAnkle": 9,
        #     "RightKnee": 10,
        #     "RightHip": 11,
        #     "RightWrist": 12,
        #     "RightElbow": 13,
        #     "RightShoulder": 14,
        #     "Head": 15
        # }
        return [FlipZ(), SwapYZ()]

    @staticmethod
    def mka_to_iisy(
            coordinate_system: bool = True,
            body_parts: bool = True
    ) -> List[Callable[[np.ndarray], np.ndarray]]:
        """Returns a list of transforms that transform positions from the MKA
        coordinate system to the IISY coordinate system as well as the MKA body
        part model to the IISY body part model.

        Note that the described coordinate system directions are described
        relative to the human actors viewing direction in the anatomical
        normal position
        (https://en.wikipedia.org/wiki/Standard_anatomical_position).
        Consequently, flipping and swapping axes may also flip the directions
        of other axes too. A solution to prevent this is:
        1. Flip all axes so that they point RIGHT, FRONT and BACK
        (or your targeted directions)
        2. Swap necessary axes

        MKA Coordinate System:
            X = Left
            Y = Down
            Z = Back

        IISY Coordinate System:
            X = Right
            Y = Front
            Z = Up
        formal transform: [0, 1, 2] -> [-0, -2, -1]

        Args:
            coordinate_system (bool): Decides whether to transform the
                coordinate system axes directions. (default=True)
            body_parts (bool): Decides whether to transform the body part model.
                (default=True)

        Returns:
            List[Callable[[np.ndarray], np.ndarray]]: List of callable
                transforms which in- and outputs np.ndarrays.
        """
        # * MKA body part model
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
        sequence_transforms = []
        if body_parts:
            sequence_transforms.append(MkaToIisyBodyParts())
        if coordinate_system:
            sequence_transforms += [FlipX(), FlipY(), FlipZ(), SwapYZ()]
        return sequence_transforms

    @staticmethod
    def hdm05_to_iisy() -> List[Callable[[np.ndarray], np.ndarray]]:
        """Returns a list of transforms that transform positions from the MKA
        coordinate system to the IISY coordinate system.

        Note that the described coordinate system directions are described
        relative to the human actors viewing direction in the anatomical
        normal position
        (https://en.wikipedia.org/wiki/Standard_anatomical_position).
        Consequently, flipping and swapping axes may also flip the directions
        of other axes too. A solution to prevent this is:
        1. Flip all axes so that they point RIGHT, FRONT and BACK
        (or your targeted directions)
        2. Swap necessary axes

        HDM05 Coordinate System:
            Original (hdm05 specification):
                X = Left
                Y = Up
                Z = Front
            After Loading with acm_asf_parser:
                X = Right
                Y = Up
                Z = Back

        IISY Coordinate System:
            X = Right
            Y = Front
            Z = Up
        formal transform [0, 1, 2] -> [0, -2, 1]

        Returns:
            List[Callable[[np.ndarray], np.ndarray]]: List of callable
                transforms which in- and outputs np.ndarrays.
        """
        # * HDM05 body part model
        # body_parts_hdm05 = {
        #     'root': 0,
        #     'lhipjoint': 1,
        #     'lfemur': 2,
        #     'ltibia': 3,
        #     'lfoot': 4,
        #     'ltoes': 5,
        #     'rhipjoint': 6,
        #     'rfemur': 7,
        #     'rtibia': 8,
        #     'rfoot': 9,
        #     'rtoes': 10,
        #     'lowerback': 11,
        #     'upperback': 12,
        #     'thorax': 13,
        #     'lowerneck': 14,
        #     'upperneck': 15,
        #     'head': 16,
        #     'lclavicle': 17,
        #     'lhumerus': 18,
        #     'lradius': 19,
        #     'lwrist': 20,
        #     'lhand': 21,
        #     'lfingers': 22,
        #     'lthumb': 23,
        #     'rclavicle': 24,
        #     'rhumerus': 25,
        #     'rradius': 26,
        #     'rwrist': 27,
        #     'rhand': 28,
        #     'rfingers': 29,
        #     'rthumb': 30
        # }
        return [FlipZ(), SwapYZ()]


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


class MkaToIisyBodyParts:
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
