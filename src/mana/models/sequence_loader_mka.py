"""An abstract loader class for specific sequence loaders."""
from mana.models.a_sequence_loader import ASequenceLoader
import json
import numpy as np
# TODO: We need to import all allowed Sequence classes
from mana.models.sequence import Sequence


class SequenceLoaderMKA(ASequenceLoader):
    """A loader for Mocap Kinect Azure motion sequences.

    Attributes:
        Transformer (Transformer): A transformer that holds transformations, that
        are applied after loading a sequence.
    """
    ALLOWED_SEQUENCE_CLASSES = [Sequence]

    def __init__(self, transformer, sequence_class):
        if sequence_class not in SequenceLoaderMKA.ALLOWED_SEQUENCE_CLASSES:
            raise ValueError(
                f'This DataLoader is not abled to load instances of '
                f'sequence_class {sequence_class}. \nAllowed sequence_classes '
                f'are {SequenceLoaderMKA.ALLOWED_SEQUENCE_CLASSES}.')
        self.sequence_class = sequence_class
        super().__init__(transformer)

    def load(self,
             path: str = None,
             json_str: str = None,
             name: str = 'Sequence MKA',
             desc=None) -> Sequence:
        """Returns a sequence represented by a MKA json file or a MKA json 
        string

        Args:
            path (str): Path to a json file representing a MKA sequence.
            json_str (str): A json string representing a MKA sequence.
            name (str): The name of the returned sequence.
            desc (str): The description of the returned sequence.
        """
        if not json_str and not path:
            raise ValueError(f'The load method expects either a file path to '
                             'a json file or a json string.')
        elif not json_str and path:
            with open(path, 'r') as sequence_file:
                json_str = sequence_file.read()
        json_data = json.loads(json_str)
        positions = np.array(json_data["positions"])

        # reshape positions to 3d array
        positions = np.reshape(
            positions,
            (np.shape(positions)[0], int(np.shape(positions)[1] / 3), 3))

        positions = super().transform(positions)
        sequence = self.sequence_class(positions, name, desc)

        return sequence

        # TODO: Implement Transformations and apply them to mka sequences when loading
        #! The Transformations to apply to transform coordinate system from MKA to IISY
        # # MKA X points left -> target X points right
        # # MKA Y points down -> target Y points front
        # # MKA Z points backwards -> target Z points up
        # # Switch Y/Z
        # y_positions_mka = positions[:, :, 1].copy()
        # z_positions_mka = positions[:, :, 2].copy()
        # positions[:, :, 1] = z_positions_mka
        # positions[:, :, 2] = y_positions_mka
        # # Flip X
        # positions[:, :, 0] *= -1
        # # Flip Y
        # positions[:, :, 1] *= -1
        # # Flip Z
        # positions[:, :, 2] *= -1

        # # NOTE: Uncomment if you want to standardize the used joint positions
        # # as in HMA
        # # Change body part indices according to the target body part format
        # # positions_mka = positions.copy()
        # # positions[:, 0, :] = positions_mka[:, 26, :]  # "head": 0
        # # positions[:, 1, :] = positions_mka[:, 3, :]  # "neck": 1
        # # positions[:, 2, :] = positions_mka[:, 5, :]  # "shoulder_l": 2
        # # positions[:, 3, :] = positions_mka[:, 12, :]  # "shoulder_r": 3
        # # positions[:, 4, :] = positions_mka[:, 6, :]  # "elbow_l": 4
        # # positions[:, 5, :] = positions_mka[:, 13, :]  # "elbow_r": 5
        # # positions[:, 6, :] = positions_mka[:, 7, :]  # "wrist_l": 6
        # # positions[:, 7, :] = positions_mka[:, 14, :]  # "wrist_r": 7
        # # positions[:, 8, :] = positions_mka[:, 1, :]  # "torso": 8 -> SpineNavel
        # # positions[:, 9, :] = positions_mka[:, 0, :]  # "pelvis": 9
        # # positions[:, 10, :] = positions_mka[:, 18, :]  # "hip_l": 10
        # # positions[:, 11, :] = positions_mka[:, 22, :]  # "hip_r": 11
        # # positions[:, 12, :] = positions_mka[:, 19, :]  # "knee_l": 12
        # # positions[:, 13, :] = positions_mka[:, 23, :]  # "knee_r": 13
        # # positions[:, 14, :] = positions_mka[:, 20, :]  # "ankle_l": 14
        # # positions[:, 15, :] = positions_mka[:, 24, :]  # "ankle_r": 15
        # # positions = positions[:, :16]