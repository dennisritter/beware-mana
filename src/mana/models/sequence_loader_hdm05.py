"""An abstract loader class for specific sequence loaders."""
from mana.models.a_sequence_loader import ASequenceLoader
import numpy as np
import mana.acm_asf_parser.amc_parser as amc_asf_parser
from mana.models.sequence import Sequence
from mana.models.sequence_transforms import SequenceTransforms


class SequenceLoaderHDM05(ASequenceLoader):
    """A loader for Mocap Kinect Azure motion sequences.

    Attributes:
        transforms (SequenceTransforms): A SequenceTransforms instance that 
        holds transformations, that are applied after loading a sequence
        (default = None).
        sequence_class (class): A class that is inherited from Sequence or the
        Sequence itself (default = Sequence).
    """
    ALLOWED_SEQUENCE_CLASSES = [Sequence]

    def __init__(self,
                 transforms: SequenceTransforms = None,
                 sequence_class=Sequence):
        if sequence_class not in SequenceLoaderHDM05.ALLOWED_SEQUENCE_CLASSES:
            raise ValueError(
                f'This DataLoader is not abled to load instances of '
                f'sequence_class {sequence_class}. \nAllowed sequence_classes '
                f'are {SequenceLoaderHDM05.ALLOWED_SEQUENCE_CLASSES}.')
        self.sequence_class = sequence_class
        super().__init__(transforms)

    def load(self,
             asf_path: str,
             amc_path: str,
             name: str = 'Sequence HDM05',
             desc=None) -> Sequence:
        """Returns a sequence loaded from HDM05 amc and asf files.

        Args:
            asf_path (str): Path to the asf file. Includes available Joints and 
            hierarchy.
            amc_path (str): Path to the amc file. Includes motions.
        """
        if not asf_path or not amc_path:
            raise ValueError(f'The load method expects a path to a ASF file and'
                             f' a path to a AMC file but at least one path were'
                             f' not given. \n[asf_path={asf_path}]'
                             f'\n[amc_path={asf_path}]')
        joints = amc_asf_parser.parse_asf(asf_path)
        motions = amc_asf_parser.parse_amc(amc_path)

        positions = []
        for frame in range(len(motions)):
            joints['root'].set_motion(motions[frame])
            positions.append(
                [joints[joint].coordinate for joint in joints.keys()])
        positions = np.array(positions).squeeze()

        sequence = self.sequence_class(positions, name=name, desc=desc)
        return sequence