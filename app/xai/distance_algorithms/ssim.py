from skimage.metrics import structural_similarity as ssim
from scipy.stats import wasserstein_distance
import numpy as np

def ssim_algorithm(maskA: np.ndarray, 
        maskB: np.ndarray) -> np.ndarray:
    """
    ...

    Args:
        maskA (np.ndarray[double]):
            A vector (like images), that can be a mask of a CAM.
        maskB (np.ndarray[double]):
            A vector (like images), that can be a mask of a CAM.
    
    Returns:
        diff (np.ndarray[double]):
            A vector showing the pixel-by-pixel differences
        score (double):
            A scalar value with the sum of the 
    """

    # maskA = np.array(maskA)
    # maskB = np.array(maskB)

    #maskA = transform_bsad(maskA)
    #maskB = transform_bsad(maskB)


    diff            = 1 - ssim(maskA, maskB, data_range=maskA.max() - maskA.min())  # full=True retorna o mapa
    # score           = wasserstein_distance(maskA.flatten(), maskB.flatten())

    return diff