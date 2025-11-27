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
    This method calculates how many standard deviations each element of diff_scores is away from the mean.

    Args:
        diff_scores (list[double]):
            Is same a modular_diff (arg of sad() method), a vector showing the pixel-by-pixel differences
            between two masks.
    
    Returns:
        units_deviation (np.ndarray[double]):
            A vector showing how many standard deviations each element is away from the mean.
        std_deviation (float):
            A scalar value with the standard deviation of the diff_scores.
    """

    mean            = np.mean(diff_scores)
    std_deviation   = np.std(diff_scores)
    units_deviation = np.abs((diff_scores - mean) / std_deviation)

    return units_deviation, std_deviation

def deviation_filter(units_deviation: np.ndarray, filter_rate: float) -> np.ndarray:
    """
    This method filters the units_deviation vector by removing a percentage of its lowest values.

    Args:
        units_deviation (np.ndarray[double]):
            A vector showing how many standard deviations each element is away from the mean.
        filter_rate (float):
            A scalar value between 0 and 1 indicating the percentage of lowest values to remove.
    
    Returns:
        units_deviation (np.ndarray[double]):
            A vector showing how many standard deviations each element is away from the mean,
            after removing the lowest values according to the filter_rate.
    """

    to_remove = round(units_deviation.size * filter_rate) 
    units_deviation = np.sort(units_deviation)

    units_deviation, removeds = units_deviation[:to_remove + 1], units_deviation[to_remove + 1:]
    for e in removeds:
        if e < 1:
            units_deviation.append(e)

    return units_deviation


