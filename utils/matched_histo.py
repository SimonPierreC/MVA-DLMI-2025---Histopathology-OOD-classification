import cv2
import numpy as np

def compute_average_histogram(images, bins=256):
    hist_accum = np.zeros((3, bins))  # Pour chaque canal (R, G, B)
    num_images = len(images)

    for img in images:
        img = (img * 255).astype(np.uint8)
        # Convert grayscale to color if needed
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        for i in range(3):  # Canaux R, G, B
            hist = cv2.calcHist([img], [i], None, [bins], [0, 256])
            hist /= hist.sum()  # Normalisation
            hist_accum[i] += hist.flatten()
    
    return hist_accum / num_images   # Moyenne des histogrammes


def match_histogram(image, reference_hist, bins=256):
    im = (image.T * 255).astype(np.uint8)
    matched = im.copy()
    for i in range(3):  # Canaux R, G, B
        orig_hist = cv2.calcHist([im], [i], None, [bins], [0, 256])
        orig_hist /= orig_hist.sum()

        cdf_orig = np.cumsum(orig_hist)  # CDF de l'image d'entrée
        cdf_ref = np.cumsum(reference_hist[i])  # CDF de la référence

        lookup_table = np.interp(cdf_orig, cdf_ref, np.arange(bins))
        matched[:, :, i] = cv2.LUT(im[:, :, i], lookup_table.astype(np.uint8))

    return matched.T / 255