"""A basic motion sequence."""
import math
import copy
import numpy as np
# TODO: Add OpenCV to Mana!
# import cv2


class Sequence:
    """Represents a motion sequence.

    Attributes:
        positions (np.ndarray): The tracked body part positions for each frame.
        name (str): The name of this sequence.
        desc (str): A description of this sequence.
    """
    def __init__(self,
                 positions: np.ndarray,
                 name: str = 'sequence',
                 desc: str = None):
        self.name = name
        # A Boolean mask list to exclude all frames, where all positions are 0.0
        zero_frames_filter_list = self._filter_zero_frames(positions)
        # Defines positions of each bodypart
        # 1. Dimension = Time/Frames
        # 2. Dimension = Bodyparts
        # 3. Dimension = x, y, z
        # Example: [
        #           [[f1_bp1_x, f1_bp1_y, f1_bp1_z], ...],
        #           [[f2_bp1_x, f2_bp1_y, f2_bp1_z], ...],
        #           ...
        #          ]
        # shape: (n_frames, n_body_parts, 3)
        self.positions = np.array(positions)[zero_frames_filter_list]
        self.desc = desc

    def __len__(self) -> int:
        """Returns the length of this sequence, which is defined by the length
        of the positions attribute.

        The sequence's length is basically the number of frames the sequence
        contains.
        """
        return len(self.positions)

    def __getitem__(self, item) -> 'Sequence':
        """Returns the sub-sequence item. You can either specifiy one element
        by index or use numpy-like slicing.

        Args:
            item (int/slice): Defines a particular frame or slice from all
            frames of this sequence.

        Raises NotImplementedError if index is given as tuple.
        Raises TypeError if item is not of type int or slice.
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
            raise TypeError(f'Invalid argument type: {type(item)}')

        return Sequence(self.positions[start:stop:step], self.name, self.desc)

    def append(self, sequence) -> 'Sequence':
        """Returns a sequence where the given sequence is concatenated to self.

        Raises ValueError if shapes of the positions property do not fit
        together.
        """
        if (self.positions.shape[1] != sequence.positions.shape[1]
                or self.positions.shape[2] != sequence.positions.shape[2]):
            raise ValueError(
                f'sequence.position shapes {self.positions.shape} and '
                f'{sequence.positions.shape} do not fit together.\nPlease '
                'ensure that the second and third axes share the same '
                'dimensionality.')

        # Copy the given sequence to not change it implicitly
        sequence = sequence[:]
        # concatenate positions
        self.positions = np.concatenate((self.positions, sequence.positions),
                                        axis=0)

        return self

    def split(self, overlap: float = 0.0, subseq_size: int = 1) -> list:
        """ Splits this sequence into batches of specified size and with
        defined overlap to each other. Returns a consecutive list of sequences
        with length of the given size.

            Args:
                overlap (float) = 0.0: How much of a batch overlaps neighboured
                batches.
                subseq_size (int) = 1: The size of the batches.
        """
        if overlap < 0.0 or overlap > 0.99:
            raise ValueError(
                'Overlap parameter must be a value between [0.0, 0.99].')
        step_size = int(subseq_size - subseq_size * overlap)
        if step_size > len(self):
            raise ValueError(
                'Step_size parameter should be smaller that this sequences '
                'length.')
        if step_size == 0:
            raise ValueError(
                'The formula int(subseq_size - subseq_size * overlap) should '
                'not equal 0. Choose params that fulfill this condition.')
        n_steps = math.floor(len(self) / step_size)
        seqs = [
            self[step * step_size:step * step_size + subseq_size]
            for step in range(0, n_steps)
        ]
        # Add batch to former name
        for i, seq in enumerate(seqs):
            seq.name = f'{seq.name}__batch-{i}'
        return seqs

    # TODO: Add OpenCV to Mana before adding this function to Sequence class
    # def to_motionimg(
    #         self,
    #         output_size=(256, 256),
    #         minmax_pos_x=(-1000, 1000),
    #         minmax_pos_y=(-1000, 1000),
    #         minmax_pos_z=(-1000, 1000),
    #         show_img=False,
    # ) -> np.ndarray:
    #     """ Returns a Motion Image, that represents this sequences' positions.

    #         Creates an Image from 3-D position data of motion sequences.
    #         Rows represent a body part (or some arbitrary position instance).
    #         Columns represent a frame of the sequence.

    #         Args:
    #             output_size (int, int): The size of the output image in pixels
    #             (height, width). Default=(200,200)
    #             minmax_pos_x (int, int): The minimum and maximum x-positions.
    #             Mapped to color range (0, 255).
    #             minmax_pos_y (int, int): The minimum and maximum y-positions.
    #             Mapped to color range (0, 255).
    #             minmax_pos_z (int, int): The minimum and maximum z-positions.
    #             Mapped to color range (0, 255).
    #     """
    #     # Create Image container
    #     img = np.zeros((len(self.positions[0, :]), len(self.positions), 3),
    #                    dtype='uint8')
    #     # 1. Map (min_pos, max_pos) range to (0, 255) Color range.
    #     # 2. Swap Axes of and frames(0) body parts(1) so rows represent body
    #     # parts and cols represent frames.
    #     img[:, :, 0] = np.interp(self.positions[:, :, 0],
    #                              [minmax_pos_x[0], minmax_pos_x[1]],
    #                              [0, 255]).swapaxes(0, 1)
    #     img[:, :, 1] = np.interp(self.positions[:, :, 1],
    #                              [minmax_pos_y[0], minmax_pos_y[1]],
    #                              [0, 255]).swapaxes(0, 1)
    #     img[:, :, 2] = np.interp(self.positions[:, :, 2],
    #                              [minmax_pos_z[0], minmax_pos_z[1]],
    #                              [0, 255]).swapaxes(0, 1)
    #     img = cv2.resize(img, output_size)

    #     if show_img:
    #         cv2.imshow(self.name, img)
    #         print(f'Showing motion image from [{self.name}]. Press any key to'
    #               ' close the image and continue.')
    #         cv2.waitKey(0)
    #         cv2.destroyAllWindows()
    #     return img

    # ? Should this function stay here?
    def _filter_zero_frames(self, positions: np.ndarray) -> list:
        """Returns a filter mask list to filter frames where all positions
        equal 0.0.

        Checks whether all coordinates for a frame are 0.0
            True -> keep this frame
            False -> remove this frame

        Args:
            positions (np.ndarray): The positions to filter
            "Zero-Position-Frames" from.

        Returns:
            (list<boolean>): The filter list.
        """
        return [len(pos) != len(pos[np.all(pos == 0)]) for pos in positions]
