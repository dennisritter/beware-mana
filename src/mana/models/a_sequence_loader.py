"""An abstract loader class for specific sequence loaders."""
from abc import ABC, ABCMeta, abstractmethod
import numpy as np
from mana.models.sequence import Sequence


class ASequenceLoader(metaclass=ABCMeta):
    """An abstract loader class for specific sequence loaders.

    Attributes:
        Transformer (Transformer): A transformer that holds transformations, that
        are applied after loading a sequence.
    """
    def __init__(self, transformer):
        self.transformer = transformer

    @abstractmethod
    def load(self) -> Sequence:
        """Abstract method to load a Sequence from some arbitrary data format 
        applying the transformations set in the given Transformer.
        """
        pass

    def transform(self, positions: 'np.ndarray'):
        """Transforms a sequence by applying the set transformations in a 
        transformer.
        """
        if self.transformer != None:
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
            for transform in self.transformer.transforms:
                trans_positions = transform(positions)
            return trans_positions
