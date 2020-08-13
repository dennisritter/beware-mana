"""An abstract loader class for specific sequence loaders."""
from abc import ABC, ABCMeta, abstractmethod
import numpy as np
from mana.models.sequence import Sequence
from mana.models.sequence_transforms import SequenceTransforms


class ASequenceLoader(metaclass=ABCMeta):
    """An abstract loader class for specific sequence loaders.

    Attributes:
        transforms (SequenceTransforms): A SequenceTransforms instance that 
        holds transformations, that are applied after loading a sequence
        (default = None).
    """
    def __init__(self, transforms: SequenceTransforms = None):
        self.transforms = transforms

    @abstractmethod
    def load(self) -> Sequence:
        """Abstract method to load a Sequence from some arbitrary data format 
        applying the transformations set in the transforms attribute.
        """
        pass

    def transform(self, positions: 'np.ndarray'):
        """Transforms a sequence by consecutively applying the 
        SequenceTransforms stored in self.transforms (a list of transforms) and 
        returns the resulting positions.

        Note that each transform in self.transforms affects the transforms 
        after it as the output of a transform is the input of the next 
        transform.
        """
        if not self.transforms:
            return positions
        else:
            if type(positions) is not np.ndarray:
                raise ValueError(
                    f'The positions parameter should be a numpy.ndarray'
                    f' but it was a {type(positions)}')
            if positions.ndim != 3:
                raise ValueError(f'The positions array should be 3-Dimensional'
                                 f'(ndim = 3) but it was {positions.ndim}')
            if positions.shape[2] != 3:
                raise ValueError(
                    f'The positions arrays shape of the last dimension'
                    f' should be 3 (shape[2] = 3) but it was '
                    f'{positions.shape[2]}.')
            trans_positions = positions
            for transform in self.transforms.transforms:
                trans_positions = transform(positions)
            return trans_positions
