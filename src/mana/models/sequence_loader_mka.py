"""An abstract loader class for specific sequence loaders."""
from mana.models.a_sequence_loader import ASequenceLoader
import json
import numpy as np
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