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
            trans_positions = positions
            for transform in self.transformer.transforms:
                trans_positions = transform(positions)
            return trans_positions
