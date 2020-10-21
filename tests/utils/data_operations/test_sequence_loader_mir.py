"""Unittests for transformation functions."""
import json
import pytest

import mana.models.sequence_transforms as st

from mana.models.sequence import Sequence
from mana.utils.data_operations.loaders.a_sequence_loader import ASequenceLoader
from mana.utils.data_operations.loaders.sequence_loader_mir import SequenceLoaderMIR  # pylint: disable=line-too-long


@pytest.fixture()
def example_sequence_transforms():
    """Returns an instance of SequenceTransforms class containing arbitrary
    transforms."""
    transforms = [st.FlipX(), st.FlipY(), st.FlipZ()]
    transform_collection = st.SequenceTransforms(transforms)
    return transform_collection


@pytest.fixture()
def sequence_json_string_mir():
    """Returns a JSON string of a MIR sequence file."""
    json_obj = {
        "name":
        "Nui-Track RealSense Data",
        "date":
        "2019-10-24T10:33:09.818504+02:00",
        "format": {
            "LeftWrist": 0,
            "LeftElbow": 1,
            "LeftShoulder": 2,
            "Neck": 3,
            "Torso": 4,
            "Waist": 5,
            "LeftAnkle": 6,
            "LeftKnee": 7,
            "LeftHip": 8,
            "RightAnkle": 9,
            "RightKnee": 10,
            "RightHip": 11,
            "RightWrist": 12,
            "RightElbow": 13,
            "RightShoulder": 14,
            "Head": 15
        },
        "timestamps": [1571905989819.5066, 1571905989835.6365],
        "positions": [
            [
                147.210175, 630.614441, 1707.68628, 260.529541, 390.415039,
                1757.9917, 210.858231, 558.265137, 1988.86121, 57.2501755,
                644.1946, 1967.00061, 34.53722, 306.513733, 2067.10913,
                23.6885338, 92.08276, 2081.747, 121.95076, -831.5958,
                2174.23853, 115.921532, -402.776245, 2095.31958, 136.45845,
                54.1958351, 2101.99316, -80.21944, -832.6035, 2202.68848,
                -71.67877, -400.948242, 2141.43823, -82.8546448, 55.910656,
                2120.34863, -207.139435, 20.50589, 2014.52136, -177.322144,
                288.219818, 2037.05713, -114.925964, 562.5209, 1967.72241,
                61.3119164, 757.557861, 1927.225
            ],
            [
                145.704437, 602.1178, 1690.16772, 268.505127, 375.29834,
                1770.69861, 207.5443, 557.670654, 1987.72583, 57.250164,
                644.195557, 1967.0542, 33.69582, 306.165955, 2067.59155,
                23.680191, 91.7240753, 2082.605, 121.094864, -831.530151,
                2174.18921, 115.834534, -402.857025, 2095.52173, 137.279236,
                53.51795, 2102.10449, -80.50644, -834.022339, 2202.70068,
                -71.88313, -402.566864, 2141.47388, -83.18657, 53.7713,
                2121.249, -207.359177, 20.7528076, 2016.41931, -178.006256,
                288.439819, 2038.654, -115.24115, 562.7535, 1968.73059,
                61.3724442, 757.176941, 1927.23633
            ],
        ]
    }
    json_str = json.dumps(json_obj)
    return json_str


def test_instance(example_sequence_transforms):  # pylint: disable=redefined-outer-name
    """Test whether we can instantiate a proper sequence loader."""
    loader = SequenceLoaderMIR(example_sequence_transforms)
    assert isinstance(loader, SequenceLoaderMIR)


def test_subclass(example_sequence_transforms):  # pylint: disable=redefined-outer-name
    """Test whether a SequenceLoaderMIR instance is a subclass of
    ASequenceLoader."""
    loader = SequenceLoaderMIR(example_sequence_transforms)
    assert issubclass(type(loader), ASequenceLoader)


def test_transforms(example_sequence_transforms):  # pylint: disable=redefined-outer-name
    """Test whether the transforms given in the constructor are available and
    equal to an instances transforms property."""
    loader = SequenceLoaderMIR(example_sequence_transforms)
    assert example_sequence_transforms == loader.transforms


def test_load(example_sequence_transforms, sequence_json_string_mir):  # pylint: disable=redefined-outer-name
    """Test whether a loaders load function returns a sequence object."""
    loader = SequenceLoaderMIR(example_sequence_transforms)
    seq = loader.load(json_str=sequence_json_string_mir)
    assert isinstance(seq, Sequence), "Return value should be 'Sequence'"
