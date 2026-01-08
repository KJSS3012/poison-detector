import numpy as np

def com_algorithm(maskA, maskB):
    def center_of_mass(cam):
        h, w = cam.shape
        xs, ys = np.meshgrid(np.arange(w), np.arange(h))
        total = cam.sum() + 1e-8
        cx = (xs * cam).sum() / total
        cy = (ys * cam).sum() / total
        return np.array([cx, cy])
    
    maskA = center_of_mass(maskA)
    maskB = center_of_mass(maskB)

    return np.linalg.norm(maskB - maskA)