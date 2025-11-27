# Sum of Absolute Differences Algorithm
import numpy as np

def sad(maskA: np.ndarray, 
        maskB: np.ndarray) -> np.ndarray, int:
    """
    Sum of Absolute Differences is an algorithm for find block matchings in two images.
    This method calculates the difference between two masks with same size.

    Args:
        maskA (np.ndarray[double]):
            A vector (like images), that can be a mask of a CAM.
        maskB (np.ndarray[double]):
            A vector (like images), that can be a mask of a CAM.
    
    Returns:
        modular_diff (np.ndarray[double]):
            A vector showing the pixel-by-pixel differences
        abs_sum (double):
            A scalar value with the sum of the modular_diff
    """

    modular_diff    = np.abs(maskA - maskB)
    abs_sum         = np.sum(modular_diff)

    return modular_diff, abs_sum

def error_margin(diff_scores: list[double]) -> np.ndarray, float:
    """
    Args:
        diff_scores (list[double]):
            Is same a modular_diff (arg of sad() method), a vector showing the pixel-by-pixel differences
            between two masks.
    
    Returns:
        units_deviation ():
        std_deviation ():
    """

    mean            = np.mean(diff_scores)
    std_deviation   = np.std(diff_scores)
    units_deviation = np.abs((diff_scores - mean) / std_deviation)

    return units_deviation, std_deviation

def deviation_filter(units_deviation: np.ndarray, filter_rate: float) -> np.ndarray:

    to_remove = round(units_deviation.size * filter_rate) 
    units_deviation = np.sort(units_deviation)

    units_deviation, removeds = units_deviation[:to_remove + 1], units_deviation[to_remove + 1:]
    for e in removeds:
        if e < 1:
            units_deviation.append(e)

    return units_deviation


