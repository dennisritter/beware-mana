import numpy as np
from mana.models.sequence import Sequence
from mana.models.sequence_loader_mka import SequenceLoaderMKA
from mana.models.transformer import Transformer


def trans1(positions):
    print("Applied Transformation 1")
    return positions


def trans2(positions):
    print("Applied Transformation 2")
    return positions


def trans3(positions):
    print("Applied Transformation 3")
    return positions


transformer = Transformer([trans1, trans2, trans3])
loader = SequenceLoaderMKA(transformer, Sequence)

mka_path = 'data/mka_samples/sample_squat_1.json'
mka_json = ''
with open(mka_path, 'r') as sequence_file:
    mka_json = sequence_file.read()
# seq = loader.load(path=mka_path, name='My loaded Sequence')
seq = loader.load(json_str=mka_json, name='My loaded Sequence')
print(seq.name)
