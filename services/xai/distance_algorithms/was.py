from scipy.stats import wasserstein_distance
import numpy as np

def wassertein_algorithm(maskA: np.ndarray, 
        maskB: np.ndarray) -> np.ndarray:
    """
    Docstring for wassertein_algorithm
    
    :param maskA: Description
    :type maskA: np.ndarray
    :param maskB: Description
    :type maskB: np.ndarray
    :return: Description
    :rtype: ndarray
    """

    score           = wasserstein_distance(maskA.flatten(), maskB.flatten())
    return score 