""" This module contains functions to compute medical meaningful angles,
convert different rotation and angle representations from and to each other.


#######ATTENTION#######
This class is not tested properly and was implemented to match a specific
use-case. Further development/ ground up refactoring is possibly needed.
#######################
"""
from enum import Enum
import warnings

import numpy as np

from mana.models.scene_graph_sequence import SceneGraphSequence
from mana.solver.a_solver import ASolver


class AngleTypes(Enum):
    """Defintion of Joint Angle Types."""
    FLEX_EX = 0  # Flexion / Extension
    AB_AD = 1  # Abduction / Adduction
    IN_EX_ROT = 2  # Internal / External rotation


def _get_joint_start_dist(joint_positions: np.ndarray) -> np.ndarray:
    """Returns the sum of distances of all frames to the starting
    x-position.

    Args:
        joint_positions (np.ndarray): The 3-D euclidean x-positions of a
            joint node.

    Returns:
        np.ndarray: The sum of all frames to the starting x-position.
    """
    return np.sum(
        np.absolute(np.absolute(joint_positions) - abs(joint_positions[0])))


def _fix_range_exceedance(joint_indices: list, joint_angles: np.ndarray,
                          threshold: int) -> np.ndarray:
    """Fixes range exceedance of joint angles that exceed the range of
    [-180, 180] degrees and returns the fixed angles.

    Medical joint angles derived from Euler sequences only support a range
    form -180 to 180 degrees. Some motions, e.g. shoulder flexions or
    abductions may result in angles that exceed this range. As a result of this
    exceedance, the calculated angle will jump to the other end of the range
    (e.g. from 180 to -180). As this issue affects other analytical parts of
    exercise evaluations, the range must be exceeded manually. Therefore, this
    function uses a defined threshold to detect abrupt changes of joint angles
    and interprets them a range exceedance. Depending on the type of exceedance
    (from -180 to 180 XOR 180 to -180) 360° degrees are added to/subtracted
    from the exceeding joint angles.

    Args:
        joint_indices (list): A list of indices, each representing a specific
            joint/body_part in a motion sequence.
        joint_angles (np.ndarray): A 3-D list of joint angles of shape
            (n_frames, n_body_parts, n_angle_types).
        threshold (int): The threshold that determines at which 1-frame angle
            change a fix is applied.

    Returns:
        np.ndarray: The fixed joint angle array.
    """
    for joint_idx in joint_indices:
        for angle_type in range(len([AngleTypes.FLEX_EX, AngleTypes.AB_AD])):
            # Get Flexion or Abduction angles for current Ball Joint
            angles = joint_angles[:, joint_idx, angle_type]
            # Get distances between angles
            distances = np.diff(angles)
            # Get indices where the absolute distance exceeds defined threshold
            jump = np.argwhere((distances > threshold)
                               | (distances < -threshold))

            # Iterate over all jump indices, with a step size of 2 as we want
            # to change a whole slice E.g.: The angle jumps from 180 to -180
            # and back to 180 later on. We want to fix all angles in between.
            for i in range(0, len(jump), 2):
                # The index in 'angles' after the jump happened
                # As len(distances) == len(angles) - 1, there is always a
                # angles[jump_idx+1] value.
                jump_post_idx = jump[i][0] + 1
                if distances[jump[i][0]] < -180:
                    # If there is another jump_idx
                    if jump[-1][0] != jump[i][0]:
                        jump_back_idx = jump[i + 1][0] + 1
                        angles[jump_post_idx:jump_back_idx + 1] += 360
                    # Else fix all angles until end
                    else:
                        angles[jump_post_idx:] += 360

                elif distances[jump[i][0]] > 180:
                    # If there is another jump_idx
                    if jump[-1][0] != jump[i][0]:
                        jump_back_idx = jump[i + 1][0]
                        angles[jump_post_idx:jump_back_idx + 1] -= 360
                    # Else fix all angles until end
                    else:
                        angles[jump_post_idx:] -= 360
            joint_angles[:, joint_idx, angle_type] = angles
    return joint_angles


def _medical_from_euler(euler_sequence: str, euler_angles: np.ndarray,
                        joint_name: str) -> np.ndarray:
    """Returns an np.array of medical joint angles in degrees seperated in
    Flexion/Extension, Abduction/Adduction and Internal/External Rotation from
    the specified euler sequence.

    Args:
        euler_sequence (str): The euler sequence to map the medical angles
            from as a string. Example: 'xyz'
        euler_angles (np.ndarray): The angles of the euler rotations about the
            corresponding axes of euler_sequence.
        joint_name (str): The name_id of the joint to retrieve medical angles.
    """
    # Determine indices of angles for roations about x,y,z axes in euler_angles
    euler_x_idx, euler_y_idx = euler_sequence.find('x'), euler_sequence.find(
        'y')

    # Init result array which will be filled and returned afterwards
    n_frames = len(euler_angles)
    n_angle_types = 3
    med_angles = np.full((n_frames, n_angle_types), None)

    if joint_name == 'shoulder_l' and euler_sequence in ['xyz', 'yxz']:
        flex = euler_angles[:, euler_x_idx]
        abd = euler_angles[:, euler_y_idx]
        # Z-rotation is meaningless at the moment, so ignore it
        med_angles[:, 0] = flex
        med_angles[:, 1] = abd
    elif joint_name == 'shoulder_r' and euler_sequence in ['xyz', 'yxz']:
        flex = euler_angles[:, euler_x_idx]
        abd = euler_angles[:, euler_y_idx] * -1
        # Z-rotation is meaningless at the moment, so ignore it
        med_angles[:, 0] = flex
        med_angles[:, 1] = abd
    elif joint_name == 'elbow_l' and euler_sequence in ['zxz']:
        flex = euler_angles[:, euler_x_idx]
        # Y-rotation is meaningless at the moment, so ignore it
        # Z-rotation is meaningless at the moment, so ignore it
        med_angles[:, 0] = flex
    elif joint_name == 'elbow_r' and euler_sequence in ['zxz']:
        flex = euler_angles[:, euler_x_idx]
        # Y-rotation is meaningless at the moment, so ignore it
        # Z-rotation is meaningless at the moment, so ignore it
        med_angles[:, 0] = flex
    elif joint_name == 'hip_l' and euler_sequence in ['xyz', 'yxz']:
        flex = euler_angles[:, euler_x_idx]
        abd = euler_angles[:, euler_y_idx]
        # Z-rotation is meaningless at the moment, so ignore it
        med_angles[:, 0] = flex
        med_angles[:, 1] = abd
    elif joint_name == 'hip_r' and euler_sequence in ['xyz', 'yxz']:
        flex = euler_angles[:, euler_x_idx]
        abd = euler_angles[:, euler_y_idx] * -1
        # Z-rotation is meaningless at the moment, so ignore it
        med_angles[:, 0] = flex
        med_angles[:, 1] = abd
    elif joint_name == 'knee_l' and euler_sequence in ['zxz']:
        flex = euler_angles[:, euler_x_idx]
        # Y-rotation is meaningless at the moment, so ignore it
        # Z-rotation is meaningless at the moment, so ignore it
        med_angles[:, 0] = flex
    elif joint_name == 'knee_r' and euler_sequence in ['zxz']:
        flex = euler_angles[:, euler_x_idx]
        # Y-rotation is meaningless at the moment, so ignore it
        # Z-rotation is meaningless at the moment, so ignore it
        med_angles[:, 0] = flex

    # If nothing has been returned yet, the specified joint wasn't supported.
    # Currently unsupported: [
    #   'head',
    #   'neck',
    #   'writst_l',
    #   'wrist_r',
    #   'torso',
    #   'pelvis',
    #   'ankle_l',
    #   'ankle_r',
    # ]
    return med_angles


class MedicalJointAngles(ASolver):
    """Computes medical meaningful joint angles."""
    def __init__(self, sequence: SceneGraphSequence) -> None:
        """
        Args:
            sequence (SceneGraphSequence): The SceneGraphSequence to analyse.
        """
        warnings.warn(
            f'The {MedicalJointAngles} module is not ready for production! '
            'It is neither thoroughly tested nor verified.'
            'Please verify that the implementation is suited to your use-case.')

        if not isinstance(sequence, SceneGraphSequence):
            raise ValueError('The MedicalJointAngle solver only supports '
                             'sequences of type SceneGraphSequence')
        super(MedicalJointAngles, self).__init__(sequence)

    def solve(self) -> np.ndarray:
        """Returns a 3-D list of joint angles for all frames, body parts and
        angle types of a sequence.

        Returns:
            np.ndarray: the array of joint angles for all frames.
        """
        n_frames = len(self.sequence.positions)
        n_body_parts = len(self.sequence.scene_graph.nodes)
        n_angle_types = 3
        body_parts = self.sequence.body_parts

        joint_angles = np.full((n_frames, n_body_parts, n_angle_types), None)

        ball_joints = ['shoulder_l', 'shoulder_r', 'hip_l', 'hip_r']
        non_ball_joints = ['elbow_l', 'elbow_r', 'knee_l', 'knee_r']

        for node in self.sequence.scene_graph.nodes:
            if 'angles' in self.sequence.scene_graph.nodes[node].keys():
                angles_dict = self.sequence.scene_graph.nodes[node]['angles']
                if node in ball_joints:
                    # TODO: Hacky/Naive/Simple solution..
                    #   ==> How can we determine order of motions more reliably?
                    # NOTE: We assume, that the axis, on which the child node
                    #       moved more gives information about whether the
                    #       current nodes' performed motion has been a flexion
                    #       followed by an abduction or vice versa.
                    #       If start_dist_x < start_dist_y, more motion occured
                    #           'frontal', which indicates a flexion->abduction
                    #           order (=> Use XYZ-Euler).
                    #       If start_dist_x > start_dist_y, more motion occured
                    #           'sideways', which indicates a abduction->flexion
                    #           order (=> Use YXZ-Euler).
                    child_node = list(
                        self.sequence.scene_graph.successors(node))[0]
                    start_dist_x = _get_joint_start_dist(
                        self.sequence.positions[:, self.sequence.
                                                body_parts[child_node], 0])
                    start_dist_y = _get_joint_start_dist(
                        self.sequence.positions[:, self.sequence.
                                                body_parts[child_node], 1])
                    if start_dist_x < start_dist_y:
                        joint_angles[:, body_parts[node]] = _medical_from_euler(
                            'xyz', angles_dict['euler_xyz'], node)
                    else:
                        joint_angles[:, body_parts[node]] = _medical_from_euler(
                            'yxz', angles_dict['euler_yxz'], node)

                elif node in non_ball_joints:
                    joint_angles[:, body_parts[node]] = _medical_from_euler(
                        'zxz', angles_dict['euler_zxz'], node)
                else:
                    joint_angles[:, body_parts[node]] = np.array(
                        [None, None, None])
        return _fix_range_exceedance([
            body_parts['shoulder_l'], body_parts['shoulder_r'],
            body_parts['hip_l'], body_parts['hip_r']
        ], joint_angles, 180)
