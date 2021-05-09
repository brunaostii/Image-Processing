import numpy as np
from sklearn.cluster import KMeans
from utils import fft_fftshift, ifft_ifftshift

def compression_kmeans(img, k):

    kmeans = KMeans(n_clusters=k)
    kmeans.fit(img)

    img = kmeans.cluster_centers_[kmeans.labels_]
    img = np.clip(img.astype('uint8'), 0, 255)

    return img

def compression_(img):
    img_comp, title = [], []
    fshift, _ = fft_fftshift(img)
    
    fft_sorted = np.sort(np.abs(fshift.reshape(-1)))

    for radius in (0.5,0.25, 0.1, 0.05, 0.01, 0.002):
        thresh = fft_sorted[int(np.floor((1 - radius) * len(fft_sorted)))]
        ind = np.abs(fshift) > thresh
        atlow = fshift * ind
    
        img_ifft = ifft_ifftshift(atlow)
        img_comp.append(img_ifft)
        title.append(radius)
   
    return img_comp, title