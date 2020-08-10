"""Represents a motion sequence whose data comes from the HDM05 motion 
    dataset in form of ASF and AMC files. This is a subclass of 
    mana.models.sequence Sequence"""
from mana.models.sequence import Sequence
import numpy as np
# ! where/how to integrate the parser?
# TODO: Add amc_asf_parser to mana!
import mana.acm_asf_parser.amc_parser as amc_asf_parser


class Sequence_hdm05_asf_amc(Sequence):
    """Represents a motion sequence whose data comes from the HDM05 motion 
    dataset in form of ASF and AMC files.

    Attributes:
        positions (np.ndarray): The tracked body part positions for each frame.
        name (str): The name of this sequence.
        desc (str): A description of this sequence.
    """
    def __init__(self,
                 positions: np.ndarray,
                 name: str = 'sequence_hdm05_asf_amc',
                 desc: str = None):
        super().__init__(self, positions, name, desc)

    @classmethod
    def from_hdm05_asf_amc_files(cls,
                                 asf_path: str,
                                 amc_path: str,
                                 name: str = 'Sequence',
                                 desc: str = None) -> 'Sequence':
        """Loads MoCap data from HDM05 amc and asf files and returns a 
        Sequence_hdm05_asf_amc object including cody part coordinates.

        Args:
            asf_path (str): Path to the asf file. Includes available Joints and 
            hierarchy.
            amc_path (str): Path to the amc file. Includes motions.

        Returns:
            Sequence_hdm05_asf_amc: a new Sequence_hdm05_asf_amc instance from 
            the given input.
        """
        joints = amc_asf_parser.parse_asf(asf_path)
        motions = amc_asf_parser.parse_amc(amc_path)

        # body_parts = {
        #     'root': 0,
        #     'lhipjoint': 1,
        #     'lfemur': 2,
        #     'ltibia': 3,
        #     'lfoot': 4,
        #     'ltoes': 5,
        #     'rhipjoint': 6,
        #     'rfemur': 7,
        #     'rtibia': 8,
        #     'rfoot': 9,
        #     'rtoes': 10,
        #     'lowerback': 11,
        #     'upperback': 12,
        #     'thorax': 13,
        #     'lowerneck': 14,
        #     'upperneck': 15,
        #     'head': 16,
        #     'lclavicle': 17,
        #     'lhumerus': 18,
        #     'lradius': 19,
        #     'lwrist': 20,
        #     'lhand': 21,
        #     'lfingers': 22,
        #     'lthumb': 23,
        #     'rclavicle': 24,
        #     'rhumerus': 25,
        #     'rradius': 26,
        #     'rwrist': 27,
        #     'rhand': 28,
        #     'rfingers': 29,
        #     'rthumb': 30
        # }

        positions = []
        for frame in range(len(motions)):
            joints['root'].set_motion(motions[frame])
            positions.append(
                [joints[joint].coordinate for joint in joints.keys()])
        positions = np.array(positions).squeeze()

        # Swap X / Y
        positions[:, :, [1, 2]] = positions[:, :, [2, 1]]
        # Negate Y
        positions[:, :, 1] = -positions[:, :, 1]
        return cls(positions, name=name, desc=desc)