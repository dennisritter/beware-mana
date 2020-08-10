"""Represents a motion sequence whose data has been recorded by the MIR 
    tracking software. This is a subclass of mana.models.sequence Sequence"""
from mana.models.sequence import Sequence
import numpy as np
import json


class Sequence_mir(Sequence):
    """Represents a motion sequence whose data has been recorded by the MIR 
    tracking software.

    Attributes:
        positions (np.ndarray): The tracked body part positions for each frame.
        name (str): The name of this sequence.
        desc (str): A description of this sequence.
    """
    def __init__(self,
                 positions: np.ndarray,
                 name: str = 'sequence_mir',
                 desc: str = None):
        super().__init__(self, positions, name, desc)

    @classmethod
    def from_mir_file(cls,
                      path: str,
                      name: str = 'Sequence',
                      desc: str = None) -> 'Sequence':
        """Loads an sequence_mir from a .json file in Mocap Intel RealSense 
        format and returns a Sequence_mir object.

        Args:
            path (str): Path to the mir json file

        Returns:
            Sequence_mir: a new Sequence_mir instance from the given input.
        """
        with open(path, 'r') as sequence_file:
            return Sequence.from_mir_json(sequence_file.read(), name)

    @classmethod
    def from_mir_json(cls,
                      json_str: str,
                      name: str = 'Sequence',
                      desc: str = None) -> 'Sequence':
        """Loads a sequence_mir from a json string in Mocap Intel RealSense 
        format and returns a Sequence_mir object.

        Args:
            json_str (str): The mir json string.

        Returns:
            Sequence_mir: a new Sequence_mir instance from the given input.
        """
        # load, parse file from json and return class
        json_data = json.loads(json_str)
        positions = np.array(json_data["positions"])

        # reshape positions to 3d array
        positions = np.reshape(
            positions,
            (np.shape(positions)[0], int(np.shape(positions)[1] / 3), 3))

        # Center Positions by subtracting the mean of each coordinate
        positions[:, :, 0] -= np.mean(positions[:, :, 0])
        positions[:, :, 1] -= np.mean(positions[:, :, 1])
        positions[:, :, 2] -= np.mean(positions[:, :, 2])

        # Adjust MIR data to our target Coordinate System
        # X_mir = Left -> X_target = Right --> Flip X-Axis
        # Y_mir = Up   -> Y_target = Front --> Switch Y and Z; Flip (new) Y-Axis
        # Z_mir = Back -> Z_target = Up    --> Switch Y and Z

        # Switch Y and Z axis.
        # In MIR Y points up and Z to the back -> We want Z to point up and Y
        # to the front,
        y_positions_mocap = positions[:, :, 1].copy()
        z_positions_mocap = positions[:, :, 2].copy()
        positions[:, :, 1] = z_positions_mocap
        positions[:, :, 2] = y_positions_mocap
        # MIR coordinate system is left handed -> flip x-axis to adjust data
        # for right handed coordinate system
        positions[:, :, 0] *= -1
        # Flip Y-Axis
        # MIR Z-Axis (our Y-Axis now) points "behind" the trainee, but we want
        # it to point "forward"
        positions[:, :, 1] *= -1

        # # Change body part indices according to the target body part format
        # positions_mocap = positions.copy()
        # positions[:, 0, :] = positions_mocap[:, 15, :]  # "head": 0
        # positions[:, 1, :] = positions_mocap[:, 3, :]  # "neck": 1
        # positions[:, 2, :] = positions_mocap[:, 2, :]  # "shoulder_l": 2
        # positions[:, 3, :] = positions_mocap[:, 14, :]  # "shoulder_r": 3
        # positions[:, 4, :] = positions_mocap[:, 1, :]  # "elbow_l": 4
        # positions[:, 5, :] = positions_mocap[:, 13, :]  # "elbow_r": 5
        # positions[:, 6, :] = positions_mocap[:, 0, :]  # "wrist_l": 6
        # positions[:, 7, :] = positions_mocap[:, 12, :]  # "wrist_r": 7
        # positions[:, 8, :] = positions_mocap[:, 4, :]  # "torso": 8
        # positions[:, 9, :] = positions_mocap[:, 5, :]  # "pelvis": 9
        # positions[:, 10, :] = positions_mocap[:, 8, :]  # "hip_l": 10
        # positions[:, 11, :] = positions_mocap[:, 11, :]  # "hip_r": 11
        # positions[:, 12, :] = positions_mocap[:, 7, :]  # "knee_l": 12
        # positions[:, 13, :] = positions_mocap[:, 10, :]  # "knee_r": 13
        # positions[:, 14, :] = positions_mocap[:, 6, :]  # "ankle_l": 14
        # positions[:, 15, :] = positions_mocap[:, 9, :]  # "ankle_r": 15

        return cls(positions, name=name, desc=desc)