import math
import numpy as np


def norm(vec: 'np.ndarray') -> 'np.ndarray':
    """Normalizes the given vector vec and returns it.

    Args:
        vec (np.ndarray): The vector to normalize.
    """
    vec_norm = np.linalg.norm(vec)
    # No zero division
    if vec_norm == 0:
        return np.zeros(len(vec))
    return vec / vec_norm