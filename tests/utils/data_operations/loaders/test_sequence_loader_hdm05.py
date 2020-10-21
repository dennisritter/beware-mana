"""Unittests for transformation functions."""
from pathlib import Path
import pytest

import mana.models.sequence_transforms as st

from mana.models.sequence import Sequence
from mana.utils.data_operations.loaders.a_sequence_loader import ASequenceLoader
from mana.utils.data_operations.loaders.sequence_loader_hdm05 import SequenceLoaderHDM05  # pylint: disable=line-too-long


@pytest.fixture()
def example_sequence_transforms():
    """Returns an instance of SequenceTransforms class containing arbitrary
    transforms."""
    transforms = [st.FlipX(), st.FlipY(), st.FlipZ()]
    transform_collection = st.SequenceTransforms(transforms)
    return transform_collection


@pytest.fixture()
def test_files_root():
    """Returns the test_data root for this test file."""
    # Return the dir, without this test file
    path = str(Path(__file__).parent).replace('\\', '/')
    path = path + '/test_data' + '/test_sequence_loader_hdm05' + '/'
    return path


def test_instance(example_sequence_transforms):  # pylint: disable=redefined-outer-name
    """Test whether we can instantiate a proper sequence loader."""
    loader = SequenceLoaderHDM05(example_sequence_transforms)
    assert isinstance(loader, SequenceLoaderHDM05)


def test_subclass(example_sequence_transforms):  # pylint: disable=redefined-outer-name
    """Test whether a SequenceLoaderhdm05 instance is a subclass of
    ASequenceLoader."""
    loader = SequenceLoaderHDM05(example_sequence_transforms)
    assert issubclass(type(loader), ASequenceLoader)


def test_transforms(example_sequence_transforms):  # pylint: disable=redefined-outer-name
    """Test whether the transforms given in the constructor are available and
    equal to an instances transforms property."""
    loader = SequenceLoaderHDM05(example_sequence_transforms)
    assert example_sequence_transforms == loader.transforms


def test_load(example_sequence_transforms, test_files_root):  # pylint: disable=redefined-outer-name
    """Test whether a loaders load function returns a sequence object."""
    path_root = str(test_files_root)
    path_asf = path_root + 'HDM_bd.asf'
    path_amc = path_root + 'HDM_bd_squat1Reps_001_120.amc'
    loader = SequenceLoaderHDM05(example_sequence_transforms)
    seq = loader.load(asf_path=path_asf, amc_path=path_amc)
    assert isinstance(seq, Sequence), "Return value should be 'Sequence'"
