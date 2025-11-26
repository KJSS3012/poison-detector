import numpy as np
import cv2
from scipy.stats import entropy
from scipy.ndimage import gaussian_filter
import glob

def get_distance_scores(mask_array, ref = None):
    def load_heatmap(path, size=(256,256)):
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE).astype(np.float32)
        img = cv2.resize(img, size)
        img = gaussian_filter(img, sigma=1.0)
        img /= img.sum() + 1e-12  # normaliza para soma=1
        return img

    def kl_divergence(p, q):
        p = np.clip(p, 1e-12, 1.0)
        q = np.clip(q, 1e-12, 1.0)
        return float(entropy(p.flatten(), q.flatten()))

    def centroid_distance(a, b):
        h, w = a.shape
        y, x = np.indices((h, w))
        def centroid(img):
            s = img.sum()
            return np.array([(y * img).sum()/s, (x * img).sum()/s])
        ca, cb = centroid(a), centroid(b)
        return float(np.linalg.norm(ca - cb))

    # exemplo
    ref = mask_array[0].cpu().detach().numpy() if ref is None else ref
    scores = []
    for img in mask_array:
        #img = load_heatmap(path)
        #kl = kl_divergence(ref, img.cpu().detach().numpy())
        #cd = centroid_distance(ref, img.cpu().detach().numpy())
        kl = kl_divergence(ref, img)
        cd = centroid_distance(ref, img)
        scores.append((np.exp(-kl) * np.exp(-cd / 10)))

    # ordenar: menor KL e menor centroid → mais parecido
    scores.sort()

    i = 1
    for score in scores:
        print(f"{i}: Score={score:.15f}")
        i += 1

    return scores[0]