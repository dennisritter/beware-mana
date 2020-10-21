"""HDM05 implementation for SequenceLoader class."""
from typing import Type

import numpy as np

import acm_asf_parser.amc_parser as amc_asf_parser
from mana.models.sequence import Sequence
from mana.models.sequence_transforms import SequenceTransforms
from mana.utils.data_operations.loaders.a_sequence_loader import \
    ASequenceLoader


class SequenceLoaderHDM05(ASequenceLoader):
    """A loader for Mocap Kinect Azure motion sequences."""
    ALLOWED_SEQUENCE_CLASSES = [Sequence]

    def __init__(self,
                 transforms: SequenceTransforms = None,
                 sequence_class: Type[Sequence] = Sequence):
        """
        Args:
            transforms (SequenceTransforms): A SequenceTransforms instance that
                holds transformations, that are applied after loading a sequence
                (default = None).
            sequence_class (Type[Sequence]): A class that is inherited from
                Sequence or the Sequence itself (default = Sequence).
        """
        if sequence_class not in SequenceLoaderHDM05.ALLOWED_SEQUENCE_CLASSES:
            raise ValueError(
                f'This DataLoader is not abled to load instances of '
                f'sequence_class {sequence_class}. \nAllowed sequence_classes '
                f'are {SequenceLoaderHDM05.ALLOWED_SEQUENCE_CLASSES}.')
        self.sequence_class = sequence_class
        super().__init__(transforms)

    def load(  # pylint: disable=arguments-differ
            self,
            asf_path: str,
            amc_path: str,
            name: str = 'Sequence HDM05',
            desc: str = None) -> Sequence:
        """Returns a sequence loaded from HDM05 amc and asf files.

        Args:
            asf_path (str): Path to the asf file. Includes available Joints and
                hierarchy.
            amc_path (str): Path to the amc file. Includes motions.
            name (str): The name of this sequence.
            desc (str): A description of this sequence.

        Returns:
            Sequence: A new loaded Sequence instance.
        """
        if not asf_path or not amc_path:
            raise ValueError(f'The load method expects a path to a ASF file and'
                             f' a path to a AMC file but at least one path were'
                             f' not given. \n[asf_path={asf_path}]'
                             f'\n[amc_path={asf_path}]')
        joints = amc_asf_parser.parse_asf(asf_path)
        motions = amc_asf_parser.parse_amc(amc_path)
        positions = []
        for frame, _ in enumerate(motions):
            joints['root'].set_motion(motions[frame])
            positions.append([joints[joint].coordinate for joint in joints])
        positions = np.array(positions).squeeze()

        positions = super().transform(positions)
        sequence = self.sequence_class(positions, name=name, desc=desc)
        return sequence
