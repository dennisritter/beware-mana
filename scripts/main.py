import numpy as np
from mana.models.sequence import Sequence
from mana.models.sequence_loader_mka import SequenceLoaderMKA
from mana.models.sequence_loader_mir import SequenceLoaderMIR
from mana.models.sequence_loader_hdm05 import SequenceLoaderHDM05
from mana.models.sequence_transforms import SequenceTransforms
import mana.utils.math.sequence_transform as st
from mana.utils.visualization.pose_visualization import vis_pose

transforms = SequenceTransforms(SequenceTransforms.hdm05_to_iisy())
# loader_original = SequenceLoaderMIR(sequence_class=Sequence)
# loader_trans = SequenceLoaderMIR(transforms=transforms, sequence_class=Sequence)
# loader_original = SequenceLoaderMKA(sequence_class=Sequence)
# loader_trans = SequenceLoaderMKA(transforms=transforms, sequence_class=Sequence)
loader_original = SequenceLoaderHDM05(sequence_class=Sequence)
loader_trans = SequenceLoaderHDM05(transforms=transforms,
                                   sequence_class=Sequence)

# path = 'data/mir_samples/191024__single__squat__user-4__4.json'
# path = 'data/mka_samples/sample_squat_1.json'
json = ''
path_asf = 'data/hdm05_samples/HDM_bd.asf'
path_amc = 'data/hdm05_samples/HDM_bd_squat1Reps_002_120.amc'
seq_o = loader_original.load(asf_path=path_asf, amc_path=path_amc)

# seq_o = loader_original.load(json_str=json)
# seq_o.positions[:, :, 0] -= np.mean(seq_o.positions[:, :, 0])
# seq_o.positions[:, :, 1] -= np.mean(seq_o.positions[:, :, 1])
# seq_o.positions[:, :, 2] -= np.mean(seq_o.positions[:, :, 2])

vis_pose(seq_o.positions[35], name='pose_o', html=True)

# seq_t = loader_trans.load(json_str=json)
# seq_t.positions[:, :, 0] -= np.mean(seq_t.positions[:, :, 0])
# seq_t.positions[:, :, 1] -= np.mean(seq_t.positions[:, :, 1])
# seq_t.positions[:, :, 2] -= np.mean(seq_t.positions[:, :, 2])

seq_t = loader_trans.load(asf_path=path_asf, amc_path=path_amc)
vis_pose(positions=seq_t.positions[35], name='pose', html=True)