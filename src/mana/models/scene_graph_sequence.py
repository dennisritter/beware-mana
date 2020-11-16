"""Contains the BewARe sequence model including the scenegraph and angle
computation."""
import copy
from typing import Union

import networkx as nx
import numpy as np
from scipy.spatial.transform import Rotation

from mana.models.sequence import Sequence
import mana.utils.math.transformations as mt


def _pelvis_coordinate_system(pelvis: np.ndarray, torso: np.ndarray,
                              hip_l: np.ndarray, hip_r: np.ndarray):
    """Returns a pelvis coordinate system defined as a tuple containing an
    origin point and a list of three normalised direction vectors.

    Constructs direction vectors that define the axes directions of the pelvis
    coordinate system.
    X-Axis-Direction:   Normalised vector whose direction points from hip_l to
        hip_r. Afterwards, it is translated so that it starts at the pelvis.
    Y-Axis-Direction:   Normalised vector whose direction is determined so that
        it is perpendicular to the hip_l-hip_r vector and points to the torso.
        Afterwards, it is translated so that it starts at the pelvis.
    Z-Axis-Direction:   The normalised cross product vector between X-Axis and
        Y-Axis that starts at the pelvis and results in a right handed
        Coordinate System.

    Args:
        pelvis (np.ndarray): The X, Y and Z coordinates of the pelvis body part.
        torso (np.ndarray): The X, Y and Z coordinates of the torso body part.
        hip_r (np.ndarray): The X, Y and Z coordinates of the hip_l body part.
        hip_l (np.ndarray): The X, Y and Z coordinates of the hip_r body part.
    """

    # Direction of hip_l -> hip_r is the direction of the X-Axis
    hip_l_hip_r = hip_r - hip_l

    # Orthogonal Projection to determine Y-Axis direction
    vec_a = torso - hip_l
    vec_b = hip_r - hip_l

    scalar = np.dot(vec_a, vec_b) / np.dot(vec_b, vec_b)
    a_on_b = (scalar * vec_b) + hip_l
    vec = torso - a_on_b

    origin = pelvis
    vec_x = mt.norm_vec(hip_l_hip_r)
    vec_z = mt.norm_vec(vec)
    vec_y = mt.orthogonal_vector(vec_z, vec_x)

    return [(origin, [vec_x, vec_y, vec_z])]


class SceneGraphSequence(Sequence):
    """Represents a motion sequence."""
    def __init__(self,
                 positions: np.ndarray,
                 scene_graph: nx.DiGraph = None,
                 body_parts: dict = None,
                 name: str = 'sequence',
                 desc: str = None):
        """
        Args:
            positions (list): The tracked body part positions for each frame.
            scene_graph (networkx.DiGraph): A Directed Graph defining the
                hierarchy between body parts that will be filled with related
                data.
            name (str): The name of this sequence.
            desc (str): A description of this sequence.
        """
        super(SceneGraphSequence, self).__init__(positions, name, desc)

        # Number, order and label of tracked body parts
        self.body_parts = {
            "head": 0,
            "neck": 1,
            "shoulder_l": 2,
            "shoulder_r": 3,
            "elbow_l": 4,
            "elbow_r": 5,
            "wrist_l": 6,
            "wrist_r": 7,
            "torso": 8,
            "pelvis": 9,
            "hip_l": 10,
            "hip_r": 11,
            "knee_l": 12,
            "knee_r": 13,
            "ankle_l": 14,
            "ankle_r": 15,
        } if body_parts is None else body_parts

        self.positions = self._get_pelvis_cs_positions(self.positions)

        if scene_graph is not None:
            self.scene_graph = scene_graph
        else:
            # A directed graph that defines the hierarchy between human body parts
            self.scene_graph = nx.DiGraph([
                ("pelvis", "torso"),
                ("torso", "neck"),
                ("neck", "head"),
                ("neck", "shoulder_l"),
                ("shoulder_l", "elbow_l"),
                ("elbow_l", "wrist_l"),
                ("neck", "shoulder_r"),
                ("shoulder_r", "elbow_r"),
                ("elbow_r", "wrist_r"),
                ("pelvis", "hip_l"),
                ("hip_l", "knee_l"),
                ("knee_l", "ankle_l"),
                ("pelvis", "hip_r"),
                ("hip_r", "knee_r"),
                ("knee_r", "ankle_r"),
            ])
            self._fill_scene_graph(self.scene_graph, self.positions)

    def __getitem__(self, item: Union[slice, tuple,
                                      int]) -> 'SceneGraphSequence':
        """Returns the sub-sequence item. You can either specifiy one element
        by index or use numpy-like slicing.

        Args:
            item Union[slice, tuple, int]: Defines a particular frame or slice
                from all frames of this sequence.

        Returns:
            SceneGraphSequence: The selected item from the Sequence as a new
                class instance.

        Raises:
            NotImplementedError: if index is given as tuple.
            TypeError: if item is not of type int or slice.
        """

        if isinstance(item, slice):
            if item.start is None and item.stop is None and item.step is None:
                # Return a Deepcopy to improve copy performance (sequence[:])
                return copy.deepcopy(self)
            start, stop, step = item.indices(len(self))
        elif isinstance(item, int):
            start, stop, step = item, item + 1, 1
        elif isinstance(item, tuple):
            raise NotImplementedError("Tuple as index")
        else:
            raise TypeError(f"Invalid argument type: {type(item)}")

        # Slice All data lists stored in the scene_graphs nodes and edges
        scene_graph = copy.deepcopy(self.scene_graph)
        for node in scene_graph.nodes:
            for vector_list in scene_graph.nodes[node][
                    'coordinate_system'].keys():
                if vector_list:
                    scene_graph.nodes[node]['coordinate_system'][
                        vector_list] = scene_graph.nodes[node][
                            'coordinate_system'][vector_list][start:stop:step]
            for angle_list in scene_graph.nodes[node]['angles'].keys():
                if angle_list:
                    scene_graph.nodes[node]['angles'][
                        angle_list] = scene_graph.nodes[node]['angles'][
                            angle_list][start:stop:step]

        return SceneGraphSequence(self.positions[start:stop:step], scene_graph,
                                  self.body_parts, self.name, self.desc)

    def append(self, sequence: 'SceneGraphSequence') -> 'SceneGraphSequence':
        """Returns the merged two sequences.

            Args:
                sequence (SceneGraphSequence): The Sequence to merge (append).

            Returns:
                SceneGraphSequence: The inplace merged sequences.

            Raises:
                ValueError: if shapes of the positions property don't fit
                    together.
            """
        super(SceneGraphSequence, self).append(sequence)

        # Copy the given sequence to not change it implicitly
        sequence = sequence[:]

        for node in sequence.scene_graph.nodes:
            merge_node_data = sequence.scene_graph.nodes[node]
            node_data = self.scene_graph.nodes[node]
            # Concatenate Coordinate System data
            node_data['coordinate_system']['origin'] = np.concatenate(
                (node_data['coordinate_system']['origin'],
                 merge_node_data['coordinate_system']['origin']))
            node_data['coordinate_system']['x_axis'] = np.concatenate(
                (node_data['coordinate_system']['x_axis'],
                 merge_node_data['coordinate_system']['x_axis']))
            node_data['coordinate_system']['y_axis'] = np.concatenate(
                (node_data['coordinate_system']['y_axis'],
                 merge_node_data['coordinate_system']['y_axis']))
            node_data['coordinate_system']['z_axis'] = np.concatenate(
                (node_data['coordinate_system']['z_axis'],
                 merge_node_data['coordinate_system']['z_axis']))

        return self

    def _get_pelvis_cs_positions(self, positions: np.ndarray) -> np.ndarray:
        """Transforms all points in positions parameter so they are relative to
        the pelvis. X-Axis = right, Y-Axis = front, Z-Axis = up.

        Args:
            positions (np.ndarray): All positions (joints) with shape ==
                (num_body_parts, num_keypoints, xyz)

        Returns:
            np.ndarray: The transformed positions relative to the pelvis.
        """
        transformed_positions = []
        for i, frame in enumerate(positions):
            transformed_positions.append([])
            pelvis_cs = _pelvis_coordinate_system(
                positions[i][self.body_parts["pelvis"]],
                positions[i][self.body_parts["torso"]],
                positions[i][self.body_parts["hip_l"]],
                positions[i][self.body_parts["hip_r"]])

            transformation = mt.projection_matrix(pelvis_cs[0][0],
                                                  pelvis_cs[0][1][0],
                                                  pelvis_cs[0][1][1])

            for _, pos in enumerate(frame):
                transformed_positions[i].append(
                    (transformation @ np.append(pos, 1))[:3])

        return np.array(transformed_positions)

    def _fill_scene_graph(self, scene_graph: nx.DiGraph,
                          positions: np.ndarray) -> None:
        """Analyses, computes transformations and eventually fills the
        scene graph.

        Args:
            scene_graph (networkx.DiGraph): A directed Graph defining the
                hierarchy between body parts that will be filled with related
                data.
            positions (np.ndarray): All positions (joints) with shape ==
                (num_body_parts, num_keypoints, xyz)
        """
        # Find Scene Graph Root Node
        root_node = None
        nodes = list(scene_graph.nodes)
        # Find root_node
        for node in nodes:
            predecessors = list(scene_graph.predecessors(node))
            if not predecessors:
                root_node = node
                break

        # Predefine node data attributes to store data for each frame of
        # the sequence
        for node in scene_graph.nodes:
            scene_graph.nodes[node]['coordinate_system'] = {}
            scene_graph.nodes[node]['angles'] = {}

        # Start recursive function with root node in our directed scene_graph
        self._calculate_transformations(scene_graph, root_node, root_node,
                                        positions)

    def _calculate_transformations(self, scene_graph: nx.DiGraph, node: str,
                                   root_node: str,
                                   positions: np.ndarray) -> None:
        """Calculates the transformations along the scene graph.

        Args:
            scene_graph (networkx.DiGraph): A directed Graph defining the
                hierarchy between body parts that will be filled with related
                data.
            node (str): The current node to compute transformation.
            root_node (str): The root node in the scene_graph.
            positions (np.ndarray): All positions (joints) with shape ==
                (num_body_parts, num_keypoints, xyz)
        """
        n_frames = len(positions)
        successors = list(scene_graph.successors(node))

        # Root Node handling
        if node == root_node:
            # The node with no predecessors is the root node, so add the
            # initial coordinate system vectors
            scene_graph.nodes[node]['coordinate_system']['origin'] = np.zeros(
                (n_frames, 3))
            x_axes = np.empty([n_frames, 3])
            x_axes[:] = np.array([1, 0, 0])
            scene_graph.nodes[node]['coordinate_system']['x_axis'] = x_axes
            y_axes = np.empty([n_frames, 3])
            y_axes[:] = np.array([0, 1, 0])
            scene_graph.nodes[node]['coordinate_system']['y_axis'] = y_axes
            z_axes = np.empty([n_frames, 3])
            z_axes[:] = np.array([0, 0, 1])
            scene_graph.nodes[node]['coordinate_system']['z_axis'] = z_axes

            # Repeat function recursive for each child node of the root node
            for child_node in successors:
                self._calculate_transformations(scene_graph, child_node,
                                                root_node, positions)
            return

        node_pos = positions[:, self.body_parts[node]]
        predecessors = list(scene_graph.predecessors(node))

        parent_node = predecessors[0]
        parent_pos = positions[:, self.body_parts[parent_node]]
        parent_cs = scene_graph.nodes[parent_node]['coordinate_system']

        # TODO: change this call to avoid the rotations input ->
        #       transformations.tranformation needs to be fixed
        # translation_mat4x4 = mt.transformation(translations=
        #       (node_pos - parent_pos))
        translation_mat4x4 = mt.transformation(
            rotations=np.tile(np.array([[[1, 0, 0], [0, 1, 0], [0, 0, 1]]]),
                              [len(node_pos), 1, 1]),
            translations=(node_pos - parent_pos))

        # If No successors or more than one successor present, add translation
        # only to (parent_node, node) edge.
        # TODO: How can we circumvent to check whether node == "torso" ?
        if len(successors) != 1 or node == "torso":
            scene_graph[parent_node][node][
                'transformation'] = translation_mat4x4
            scene_graph.nodes[node]['coordinate_system']["origin"] = mt.bmvm(
                translation_mat4x4, mt.v3_to_v4(parent_cs['origin']))[:, :3]

            scene_graph.nodes[node]['coordinate_system']["x_axis"] = parent_cs[
                'x_axis']
            scene_graph.nodes[node]['coordinate_system']["y_axis"] = parent_cs[
                'y_axis']
            scene_graph.nodes[node]['coordinate_system']["z_axis"] = parent_cs[
                'z_axis']
        elif len(successors) == 1:
            child_node = successors[0]
            child_pos = positions[:, self.body_parts[child_node]]

            # --Determine Joint Rotation--
            scene_graph.nodes[node]['angles'] = {}
            # Get direction vector from node to child to determine rotation of
            # nodes' joint
            node_to_child_node = mt.norm_vec(child_pos - node_pos)

            # Get parent coordinate system Z-axis as reference for nodes'
            # joint rotation..
            # ..Determine 4x4 homogenious rotation matrix to derive joint
            # angles later
            #! orthogonal vector computation is different in old transform &
            #       mana.transform
            rot_parent_to_node = mt.rotation_from_vectors(
                parent_cs['z_axis'] * -1, node_to_child_node)
            rot_parent_to_node = mt.transformation(
                rotations=rot_parent_to_node,
                translations=np.tile(np.array([[0, 0, 0]]),
                                     [len(rot_parent_to_node), 1]))

            # Get Euler Sequences to be able to determine medical joint angles
            # NOTE: Scipy from_dcm() function has been renamed to 'as_matrix()'
            #       in scipy=1.4.* - lates version for win64 is still 1.3.* ;
            #       Consider updating scipy dependency when 1.4.* is available
            #       for win64.
            euler_angles_xyz = Rotation.from_dcm(
                rot_parent_to_node[:, :3, :3]).as_euler('XYZ', degrees=True)
            euler_angles_yxz = Rotation.from_dcm(
                rot_parent_to_node[:, :3, :3]).as_euler('YXZ', degrees=True)
            euler_angles_zxz = Rotation.from_dcm(
                rot_parent_to_node[:, :3, :3]).as_euler('ZXZ', degrees=True)

            # Store Euler Sequences
            scene_graph.nodes[node]['angles']['euler_xyz'] = euler_angles_xyz
            scene_graph.nodes[node]['angles']['euler_yxz'] = euler_angles_yxz
            scene_graph.nodes[node]['angles']['euler_zxz'] = euler_angles_zxz
            # Store the nodes coordinate system
            scene_graph.nodes[node]['coordinate_system']['origin'] = mt.bmvm(
                translation_mat4x4, mt.v3_to_v4(parent_cs['origin']))[:, :3]

            x_axes = mt.norm_vec(
                mt.bmvm(rot_parent_to_node[:, :3, :3], parent_cs['x_axis']))
            scene_graph.nodes[node]['coordinate_system']['x_axis'] = x_axes

            y_axes = mt.norm_vec(
                mt.bmvm(rot_parent_to_node[:, :3, :3], parent_cs['y_axis']))
            scene_graph.nodes[node]['coordinate_system']['y_axis'] = y_axes

            z_axes = mt.norm_vec(
                mt.bmvm(rot_parent_to_node[:, :3, :3], parent_cs['z_axis']))

            scene_graph.nodes[node]['coordinate_system']['z_axis'] = z_axes

        # Repeat procedure if successors present
        for child_node in successors:
            self._calculate_transformations(scene_graph, child_node, root_node,
                                            positions)

        return
