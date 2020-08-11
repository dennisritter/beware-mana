"""An abstract loader class for specific sequence loaders."""
from mana.models.a_sequence_loader import ASequenceLoader
import json
import numpy as np
# TODO: We need to import all allowed Sequence classes
from mana.models.sequence import Sequence


class SequenceLoaderMIR(ASequenceLoader):
    """A loader for Mocap Kinect Azure motion sequences.

    Attributes:
        Transformer (Transformer): A transformer that holds transformations, that
        are applied after loading a sequence.
    """
    ALLOWED_SEQUENCE_CLASSES = [Sequence]

    def __init__(self, transformer, sequence_class):
        if sequence_class not in SequenceLoaderMIR.ALLOWED_SEQUENCE_CLASSES:
            raise ValueError(
                f'This DataLoader is not abled to load instances of '
                f'sequence_class {sequence_class}. \nAllowed sequence_classes '
                f'are {SequenceLoaderMIR.ALLOWED_SEQUENCE_CLASSES}.')
        self.sequence_class = sequence_class
        super().__init__(transformer)

    def load(self,
             path: str = None,
             json_str: str = None,
             name: str = 'Sequence MIR',
             desc=None) -> Sequence:
        """Returns a sequence represented by a MIR json file or a MIR json 
        string

        Args:
            path (str): Path to a json file representing a MIR sequence.
            json_str (str): A json string representing a MIR sequence.
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
        #! The Transformations to apply to transform coordinate system from MIR to IISY
        # # Center Positions by subtracting the mean of each coordinate
        # positions[:, :, 0] -= np.mean(positions[:, :, 0])
        # positions[:, :, 1] -= np.mean(positions[:, :, 1])
        # positions[:, :, 2] -= np.mean(positions[:, :, 2])

        # # Adjust MIR data to our target Coordinate System
        # # X_mir = Left -> X_target = Right --> Flip X-Axis
        # # Y_mir = Up   -> Y_target = Front --> Switch Y and Z; Flip (new) Y-Axis
        # # Z_mir = Back -> Z_target = Up    --> Switch Y and Z

        # # Switch Y and Z axis.
        # # In MIR Y points up and Z to the back -> We want Z to point up and Y
        # # to the front,
        # y_positions_mocap = positions[:, :, 1].copy()
        # z_positions_mocap = positions[:, :, 2].copy()
        # positions[:, :, 1] = z_positions_mocap
        # positions[:, :, 2] = y_positions_mocap
        # # MIR coordinate system is left handed -> flip x-axis to adjust data
        # # for right handed coordinate system
        # positions[:, :, 0] *= -1
        # # Flip Y-Axis
        # # MIR Z-Axis (our Y-Axis now) points "behind" the trainee, but we want
        # # it to point "forward"
        # positions[:, :, 1] *= -1

        # # # Change body part indices according to the target body part format
        # # positions_mocap = positions.copy()
        # # positions[:, 0, :] = positions_mocap[:, 15, :]  # "head": 0
        # # positions[:, 1, :] = positions_mocap[:, 3, :]  # "neck": 1
        # # positions[:, 2, :] = positions_mocap[:, 2, :]  # "shoulder_l": 2
        # # positions[:, 3, :] = positions_mocap[:, 14, :]  # "shoulder_r": 3
        # # positions[:, 4, :] = positions_mocap[:, 1, :]  # "elbow_l": 4
        # # positions[:, 5, :] = positions_mocap[:, 13, :]  # "elbow_r": 5
        # # positions[:, 6, :] = positions_mocap[:, 0, :]  # "wrist_l": 6
        # # positions[:, 7, :] = positions_mocap[:, 12, :]  # "wrist_r": 7
        # # positions[:, 8, :] = positions_mocap[:, 4, :]  # "torso": 8
        # # positions[:, 9, :] = positions_mocap[:, 5, :]  # "pelvis": 9
        # # positions[:, 10, :] = positions_mocap[:, 8, :]  # "hip_l": 10
        # # positions[:, 11, :] = positions_mocap[:, 11, :]  # "hip_r": 11
        # # positions[:, 12, :] = positions_mocap[:, 7, :]  # "knee_l": 12
        # # positions[:, 13, :] = positions_mocap[:, 10, :]  # "knee_r": 13
        # # positions[:, 14, :] = positions_mocap[:, 6, :]  # "ankle_l": 14
        # # positions[:, 15, :] = positions_mocap[:, 9, :]  # "ankle_r": 15