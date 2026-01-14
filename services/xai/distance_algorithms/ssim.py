from skimage.metrics import structural_similarity as ssim
import numpy as np

def is_constant(mask: np.ndarray) -> bool:
    return np.all(mask == mask.flat[0])

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
    # Caso 1: ambas nulas (ou constantes iguais)
    if is_constant(maskA) and is_constant(maskB):
        if np.all(maskA == maskB):
            return 0.0  # nenhuma diferença
        else:
            return 1.0  # constantes diferentes

    # Caso 2: apenas uma é nula/constante
    if is_constant(maskA) or is_constant(maskB):
        return 1.0  # máxima diferença

    # Caso normal
    score = ssim(
        maskA,
        maskB,
        data_range=maskA.max() - maskA.min()
    )

    diff = 1 - score
    return diff