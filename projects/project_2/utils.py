import numpy as np
import matplotlib.pyplot as plt 
from cv2 import cv2

def fft_fftshift(img):
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)  #Deslocamento do componente de frequência-zero para o centro do espectro
    magnitude_spectrum = 20*np.log(np.abs(fshift))

    return fshift, magnitude_spectrum

def ifft_ifftshift(img, origin_min=0, origin_max=255):
    f_ishift = np.fft.ifftshift(img) # Retornando a componente de frequência-zero para o lugar original
    img_back = np.fft.ifft2(f_ishift)
    img_back = np.abs(img_back)
    norm_image = cv2.normalize(img_back, None, alpha=origin_min, beta=origin_max, norm_type=cv2.NORM_MINMAX, dtype=cv2.DFT_COMPLEX_OUTPUT).astype(np.uint8)
    return img_back

def rotation_45(img, angle=45):
    rows,cols = img.shape

    M = cv2.getRotationMatrix2D((cols/2,rows/2),angle,0.7)
    dst = cv2.warpAffine(np.float32(img),M,(cols,rows))

    return dst

def plot_(img, magnitude_spectrum, img_back, title):
    plt.figure(figsize=(20,8))
    plt.subplot(131),plt.imshow(img, cmap = 'gray')
    plt.title(title[0]), plt.xticks([]), plt.yticks([])
    plt.subplot(132),plt.imshow(magnitude_spectrum, cmap = 'gray')
    plt.title(title[1]), plt.xticks([]), plt.yticks([])
    plt.subplot(133),plt.imshow(img_back, cmap = 'gray')
    plt.title(title[2]), plt.xticks([]), plt.yticks([])
    plt.suptitle(title[3], fontsize=14)
    plt.savefig(title[3]+'.png')
    plt.show()

def plot_6(imgs, title, subtitle):
    plt.figure(figsize=(20,8))
    plt.subplot(231),plt.imshow(imgs[0], cmap = 'gray')
    plt.title(title[0]), plt.xticks([]), plt.yticks([])

    plt.subplot(232),plt.imshow(imgs[1], cmap = 'gray')
    plt.title(title[1]), plt.xticks([]), plt.yticks([])

    plt.subplot(233),plt.imshow(imgs[2], cmap = 'gray')
    plt.title(title[2]), plt.xticks([]), plt.yticks([])

    plt.subplot(234),plt.imshow(imgs[3], cmap = 'gray')
    plt.title(title[3]), plt.xticks([]), plt.yticks([])

    plt.subplot(235),plt.imshow(imgs[4], cmap = 'gray')
    plt.title(title[4]), plt.xticks([]), plt.yticks([])

    plt.subplot(236),plt.imshow(imgs[5], cmap = 'gray')
    plt.title(title[5]), plt.xticks([]), plt.yticks([])

    plt.suptitle(subtitle, fontsize=14)
    plt.savefig(subtitle+'.png')
    plt.show()