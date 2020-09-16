import numpy as np
from mana.models.sequence import Sequence
from mana.utils.data_operations.loaders.sequence_loader_mka import SequenceLoaderMKA
from mana.models.sequence_transforms import SequenceTransforms
# import mana.utils.math.sequence_transform as st
from mana.utils.visualization.pose_visualization import vis_pose

transforms = SequenceTransforms(SequenceTransforms.mka_to_iisy())
loader_trans = SequenceLoaderMKA(transforms=transforms, sequence_class=Sequence)

path = 'data/mka_samples/sample_squat_1.json'

seq_t = loader_trans.load(path=path)

seq_t.positions[:, :, 0] -= np.mean(seq_t.positions[:, :, 0])
seq_t.positions[:, :, 1] -= np.mean(seq_t.positions[:, :, 1])
seq_t.positions[:, :, 2] -= np.mean(seq_t.positions[:, :, 2])

vis_pose(positions=seq_t.positions[35], name='pose', html=True)
print(seq_t.positions.shape)
