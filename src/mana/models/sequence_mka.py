"""Represents a motion sequence whose data has been recorded by the MKA 
    tracking software. This is a subclass of mana.models.sequence Sequence"""
from mana.models.sequence import Sequence
import numpy as np
import json


class Sequence_mka(Sequence):
    """Represents a motion sequence whose data has been recorded by the MKA 
    tracking software.

    Attributes:
        positions (np.ndarray): The tracked body part positions for each frame.
        name (str): The name of this sequence.
        desc (str): A description of this sequence.
    """
    def __init__(self,
                 positions: np.ndarray,
                 name: str = 'sequence_mka',
                 desc: str = None):
        super().__init__(self, positions, name, desc)

    @classmethod
    def from_mka_file(cls,
                      path: str,
                      name: str = 'Sequence',
                      desc: str = None) -> 'Sequence':
        """Loads a sequence_mka from a .json file in Mocap Kinect Azure format and 
        returns a Sequence_mka object.

        Args:
            path (str): Path to the mka json file

        Returns:
            Sequence_mka: a new Sequence_mka instance from the given input.
        """
        with open(path, 'r') as sequence_file:
            return Sequence.from_mka_json(sequence_file.read(), name, desc)

    @classmethod
    def from_mka_json(cls,
                      json_str: str,
                      name: str = 'Sequence',
                      desc: str = None) -> 'Sequence':
        """Loads a sequence_mka from a json string in Mocap Kinect Azure 
        format and returns an Sequence object.

        Args:
            json_str (str): The mka json string.

        Returns:
            Sequence_mka: a new Sequence_mka instance from the given input.
        """
        # load, parse file from json and return class
        json_data = json.loads(json_str)
        positions = np.array(json_data["positions"])

        # reshape positions to 3d array
        positions = np.reshape(
            positions,
            (np.shape(positions)[0], int(np.shape(positions)[1] / 3), 3))

        # MKA X points left -> target X points right
        # MKA Y points down -> target Y points front
        # MKA Z points backwards -> target Z points up
        # Switch Y/Z
        y_positions_mka = positions[:, :, 1].copy()
        z_positions_mka = positions[:, :, 2].copy()
        positions[:, :, 1] = z_positions_mka
        positions[:, :, 2] = y_positions_mka
        # Flip X
        positions[:, :, 0] *= -1
        # Flip Y
        positions[:, :, 1] *= -1
        # Flip Z
        positions[:, :, 2] *= -1

        # NOTE: Uncomment if you want to standardize the used joint positions
        # as in HMA
        # Change body part indices according to the target body part format
        # positions_mka = positions.copy()
        # positions[:, 0, :] = positions_mka[:, 26, :]  # "head": 0
        # positions[:, 1, :] = positions_mka[:, 3, :]  # "neck": 1
        # positions[:, 2, :] = positions_mka[:, 5, :]  # "shoulder_l": 2
        # positions[:, 3, :] = positions_mka[:, 12, :]  # "shoulder_r": 3
        # positions[:, 4, :] = positions_mka[:, 6, :]  # "elbow_l": 4
        # positions[:, 5, :] = positions_mka[:, 13, :]  # "elbow_r": 5
        # positions[:, 6, :] = positions_mka[:, 7, :]  # "wrist_l": 6
        # positions[:, 7, :] = positions_mka[:, 14, :]  # "wrist_r": 7
        # positions[:, 8, :] = positions_mka[:, 1, :]  # "torso": 8 -> SpineNavel
        # positions[:, 9, :] = positions_mka[:, 0, :]  # "pelvis": 9
        # positions[:, 10, :] = positions_mka[:, 18, :]  # "hip_l": 10
        # positions[:, 11, :] = positions_mka[:, 22, :]  # "hip_r": 11
        # positions[:, 12, :] = positions_mka[:, 19, :]  # "knee_l": 12
        # positions[:, 13, :] = positions_mka[:, 23, :]  # "knee_r": 13
        # positions[:, 14, :] = positions_mka[:, 20, :]  # "ankle_l": 14
        # positions[:, 15, :] = positions_mka[:, 24, :]  # "ankle_r": 15
        # positions = positions[:, :16]

        return cls(positions, name=name, desc=desc)