import numpy as np

def interquartile_range_treshold(data: list[float], factor: float = 1) -> float:
    """
    This method calculates the interquartile range (IQR) of the given data and returns a treshold value.
    The treshold is calculated as Q3 + factor * IQR, where Q3 is the third quartile and IQR is the interquartile range.

    Args:
        data (list[float]):
            A list of numerical values. 
        factor (float):
            A multiplier for the IQR to determine the treshold. Default is 1.5
    Returns:
        treshold (float):
            A scalar value representing the calculated treshold.
    """
    q1 = np.percentile(data, 25)
    q3 = np.percentile(data, 75)
    iqr = q3 - q1
    treshold = q3 + factor * iqr
    return treshold