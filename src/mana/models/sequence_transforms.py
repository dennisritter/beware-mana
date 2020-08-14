"""A collection class for SequenceTransforms."""
import mana.utils.math.sequence_transform as t


class SequenceTransforms:
    """A collection class for SequenceTransforms."""
    def __init__(self, transforms: list):
        self.transforms = transforms

    # * IISY body part model
    #     body_parts = {
    #     "head": 0,
    #     "neck": 1,
    #     "shoulder_l": 2,
    #     "shoulder_r": 3,
    #     "elbow_l": 4,
    #     "elbow_r": 5,
    #     "wrist_l": 6,
    #     "wrist_r": 7,
    #     "torso": 8,
    #     "pelvis": 9,
    #     "hip_l": 10,
    #     "hip_r": 11,
    #     "knee_l": 12,
    #     "knee_r": 13,
    #     "ankle_l": 14,
    #     "ankle_r": 15,
    # }
    @staticmethod
    def mir_to_iisy():
        """Returns a list of transforms that transform positions from the MIR
        coordinate system to the IISY coordinate system.

        Note that the described coordinate system directions are described
        relative to the human actors viewing direction in the anatomical
        normal position
        (https://en.wikipedia.org/wiki/Standard_anatomical_position).
        Conseuqently, flipping and swapping axes may also flip the directions
        of other axes too. A solution to prevent this is:
        1. Flip all axes so that they point RIGHT, FRONT and BACK
        (or your targeted directions)
        2. Swap necessary axes

        MIR Coordinate System:
            X = Right
            Y = Up
            Z = Back

        IISY Coordinate System:
            X = Right
            Y = Front
            Z = Up
        formal transform: [0, 1, 2] -> [0, -2, 1]
        """
        # * MIR body part model
        # body_parts_mir = {
        #     "LeftWrist": 0,
        #     "LeftElbow": 1,
        #     "LeftShoulder": 2,
        #     "Neck": 3,
        #     "Torso": 4,
        #     "Waist": 5,
        #     "LeftAnkle": 6,
        #     "LeftKnee": 7,
        #     "LeftHip": 8,
        #     "RightAnkle": 9,
        #     "RightKnee": 10,
        #     "RightHip": 11,
        #     "RightWrist": 12,
        #     "RightElbow": 13,
        #     "RightShoulder": 14,
        #     "Head": 15
        # }
        return [t.FlipZ(), t.SwapYZ()]

    @staticmethod
    def mka_to_iisy():
        """Returns a list of transforms that transform positions from the MKA
        coordinate system to the IISY coordinate system.

        Note that the described coordinate system directions are described
        relative to the human actors viewing direction in the anatomical
        normal position
        (https://en.wikipedia.org/wiki/Standard_anatomical_position).
        Conseuqently, flipping and swapping axes may also flip the directions
        of other axes too. A solution to prevent this is:
        1. Flip all axes so that they point RIGHT, FRONT and BACK
        (or your targeted directions)
        2. Swap necessary axes

        MKA Coordinate System:
            X = Left
            Y = Down
            Z = Back

        IISY Coordinate System:
            X = Right
            Y = Front
            Z = Up
        formal transform: [0, 1, 2] -> [-0, -2, -1]
        """
        # * MKA body part model
        # body_parts_mka = {
        #     "Pelvis": 0,
        #     "SpineNavel": 1,
        #     "SpineChest ": 2,
        #     "Neck": 3,
        #     "ClavicleLeft": 4,
        #     "ShoulderLeft": 5,
        #     "ElbowLeft ": 6,
        #     "WristLeft ": 7,
        #     "HandLeft ": 8,
        #     "HandTipLeft": 9,
        #     "ThumbLeft": 10,
        #     "ClavicleRight": 11,
        #     "ShoulderRight": 12,
        #     "ElbowRight": 13,
        #     "WristRight": 14,
        #     "HandRight": 15,
        #     "HandTipRight": 16,
        #     "ThumbRight": 17,
        #     "HipLeft": 18,
        #     "KneeLeft": 19,
        #     "AnkleLeft": 20,
        #     "FootLeft": 21,
        #     "HipRight": 22,
        #     "KneeRight": 23,
        #     "AnkleRight": 24,
        #     "FootRight": 25,
        #     "Head": 26,
        #     "Nose": 27,
        #     "EyeLeft": 28,
        #     "EarLeft": 29,
        #     "EyeRight": 30,
        #     "EarRight": 31
        # }
        return [t.FlipX(), t.FlipY(), t.FlipZ(), t.SwapYZ()]

    @staticmethod
    def hdm05_to_iisy():
        """Returns a list of transforms that transform positions from the MKA
        coordinate system to the IISY coordinate system.

        Note that the described coordinate system directions are described
        relative to the human actors viewing direction in the anatomical
        normal position
        (https://en.wikipedia.org/wiki/Standard_anatomical_position).
        Conseuqently, flipping and swapping axes may also flip the directions
        of other axes too. A solution to prevent this is:
        1. Flip all axes so that they point RIGHT, FRONT and BACK
        (or your targeted directions)
        2. Swap necessary axes

        HDM05 Coordinate System:
            Original (hdm05 specification):
                X = Left
                Y = Up
                Z = Front
            After Loading with acm_asf_parser:
                X = Right
                Y = Up
                Z = Back

        IISY Coordinate System:
            X = Right
            Y = Front
            Z = Up
        formal transform [0, 1, 2] -> [0, -2, 1]
        """
        # * HDM05 body part model
        # body_parts_hdm05 = {
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
        return [t.FlipZ(), t.SwapYZ()]
