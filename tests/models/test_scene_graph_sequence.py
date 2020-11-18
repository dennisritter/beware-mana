"""Unittests for SceneGraphSequence transforms."""
import numpy as np
import pytest

from mana.models.scene_graph_sequence import SceneGraphSequence


@pytest.fixture()
def ground_truth():
    """Returns a numpy array of positions as they are expected by the
    SceneGraphSequence constructor. (3D = [frame, body part, xyz])
    """
    json_data = {
        "name":
        "squat",
        "date":
        "2020-08-19T11:58:10.0253572+02:00",
        "format": {
            "Pelvis": 0,
            "SpineNavel": 1,
            "SpineChest": 2,
            "Neck": 3,
            "ClavicleLeft": 4,
            "ShoulderLeft": 5,
            "ElbowLeft": 6,
            "WristLeft": 7,
            "HandLeft": 8,
            "HandTipLeft": 9,
            "ThumbLeft": 10,
            "ClavicleRight": 11,
            "ShoulderRight": 12,
            "ElbowRight": 13,
            "WristRight": 14,
            "HandRight": 15,
            "HandTipRight": 16,
            "ThumbRight": 17,
            "HipLeft": 18,
            "KneeLeft": 19,
            "AnkleLeft": 20,
            "FootLeft": 21,
            "HipRight": 22,
            "KneeRight": 23,
            "AnkleRight": 24,
            "FootRight": 25,
            "Head": 26,
            "Nose": 27,
            "EyeLeft": 28,
            "EarLeft": 29,
            "EyeRight": 30,
            "EarRight": 31
        },
        "timestamps": [1597831088336, 1597831088367],
        "positions": [
            [
                112.493462, 59.1248474, 2466.08447, 120.905479, -135.81868,
                2416.88037, 127.63205, -291.662872, 2377.52954, 134.431732,
                -535.9141, 2393.761, 170.569626, -494.355621, 2398.613,
                329.299438, -496.681335, 2370.34766, 417.7305, -445.120819,
                2075.93481, 426.0511, -542.594238, 1833.3009, 406.867035,
                -520.49176, 1728.70459, 385.911316, -529.1491, 1608.9519,
                360.989, -505.034515, 1695.50208, 95.76902, -496.608124,
                2385.89575, -34.91211, -517.5675, 2318.90918, -96.34474,
                -438.149872, 2017.59985, -16.6398888, -463.917725, 1766.26648,
                11.41394, -554.497253, 1697.82043, 46.3887062, -530.998535,
                1585.6748, 68.95936, -494.378723, 1634.65979, 214.676346,
                60.6087, 2477.675, 284.407928, 490.5376, 2358.49365, 314.4065,
                919.6264, 2393.41016, 351.647, 1036.637, 2221.44067, 20.35065,
                57.78679, 2455.63281, -25.2869282, 474.934631, 2290.18872,
                -47.88162, 902.3099, 2373.94434, -91.70903, 999.347534,
                2211.27783, 139.512939, -627.4859, 2384.907, 161.552322,
                -689.382935, 2215.9856, 187.251282, -723.276855, 2262.64,
                236.367554, -694.634766, 2393.84253, 129.402313, -723.0471,
                2251.18066, 50.52366, -706.0563, 2374.99487
            ],
            [
                116.11795, 68.53342, 2474.74951, 122.91555, -126.688805,
                2425.10938, 128.962448, -281.339172, 2380.2854, 132.9814,
                -526.532166, 2384.88159, 169.629517, -485.600433,
                2391.78955, 327.9933, -493.19043, 2361.05664, 418.779938,
                -440.5948, 2067.02539, 422.660278, -526.348, 1819.4613,
                408.299957, -504.47403, 1713.87073, 383.150879, -511.870972,
                1594.64709, 357.740448, -493.8135,
                1684.04089, 94.72279, -486.397, 2378.81274, -34.589962,
                -506.698059, 2308.53271, -89.5071945, -434.2509, 2003.70313,
                -8.465558, -458.4431, 1752.2019, 37.2206459, -526.4446,
                1668.48755, 53.6750755, -496.270782, 1553.52686, 81.164444,
                -484.6796, 1626.81616, 218.3893, 68.96458, 2487.05859,
                289.411224, 492.8208, 2346.175, 315.7543, 922.1856, 2388.28882,
                350.8903, 1038.82446, 2215.21948, 23.8953686, 68.1446152,
                2463.64966, -30.22573, 474.803925, 2274.42651, -47.49163,
                899.799255, 2373.69287, -93.6681061, 1002.46466, 2214.80957,
                137.061188, -617.773254, 2371.664, 164.187866, -670.8532,
                2200.1958, 188.407135, -707.3109, 2245.83228, 233.6029,
                -685.769836, 2380.01318, 130.847763, -706.2588, 2232.61768,
                48.11064, -695.4994, 2354.87964
            ]
        ],
    }
    positions = np.array(json_data["positions"])

    # reshape positions to 3d array
    json_data['positions'] = np.reshape(
        positions, (np.shape(positions)[0], int(np.shape(positions)[1] / 3), 3))

    return json_data


def test_instance(ground_truth):
    """Test whether a SceneGraphSequence object is instantiated."""
    seq = SceneGraphSequence(ground_truth['positions'])
    assert isinstance(seq, SceneGraphSequence)


def test_len(ground_truth):
    """Test whether the len function returns the correct length."""
    seq = SceneGraphSequence(ground_truth['positions'])
    assert len(seq) == 2


def test_append(ground_truth):
    """Test whether the append function returns the correct positions."""
    seq1 = SceneGraphSequence(ground_truth['positions'])
    seq2 = SceneGraphSequence(ground_truth['positions'])
    seq1.append(seq2)
    np.testing.assert_array_equal(
        seq1.positions, np.concatenate((seq2.positions, seq2.positions)))


def test_split(ground_truth):
    """Test whether the split function splits the SceneGraphSequence in correct chunks."""
    seq = SceneGraphSequence(ground_truth['positions'])
    split1 = seq.split(overlap=0.0, subseq_size=1)
    assert len(split1) == 2
    # Check if ValueError for overlap >= 1
    with pytest.raises(ValueError):
        seq.split(overlap=1, subseq_size=1)
    # Check if ValueError for step_size > len(seq)
    with pytest.raises(ValueError):
        seq.split(overlap=0.0, subseq_size=3)
    # Check if ValueError for When resulting step_size is == 0
    with pytest.raises(ValueError):
        seq.split(overlap=0.0, subseq_size=0)
    # Check if ValueError for When resulting step_size is < 0
    with pytest.raises(ValueError):
        seq.split(overlap=0.0, subseq_size=-2)
