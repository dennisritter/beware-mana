"""An abstract loader class for specific sequence loaders."""
from abc import ABCMeta, abstractmethod
from typing import Any

from mana.models.sequence import Sequence


class ASolver(metaclass=ABCMeta):
    """An abstract loader class for specific sequence loaders."""
    def __init__(self, sequence: Sequence = None):
        """
        Args:
            sequence (Sequence): A Sequence instance that is used by the solver.
                (default = None).
        """
        self.sequence = sequence

    @abstractmethod
    def solve(self) -> Any:
        """Abstract method to start the solver computation.

        Returns:
            Any: The solvers result.
        """
