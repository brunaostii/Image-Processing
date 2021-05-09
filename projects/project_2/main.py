from cv2 import cv2
import matplotlib.pyplot as plt
import argparse
from filters import *
from utils import *
from compression import *


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Project 2 - Frequency filtering')
    parser.add_argument('-i', type=str, help='image input', required=True)
    parser.add_argument('-o', type=str, help='image output', required=True)
    parser.add_argument('-f', type=str, help='c) colour, m) monochromatic', required=True)
    args = parser.parse_args()

    img = cv2.imread('../../images/bw/baboon.png', cv2.IMREAD_GRAYSCALE)

    if args.f == 'c':
        print('Only monochromatic images!')
        exit()

    # Original Image, Magnitude Spectrum and IFFT -> original image
    fshift_before, magnitude_spectrum = fft_fftshift(img)
    img_ifft = ifft_ifftshift(fshift_before)
    plot_(img, magnitude_spectrum, img_ifft, ['Imagem original (512, 512 pixels)', 'espectro de Fourier (magnitude)', 'imagem após inversa de Fourier', 'imagens originais'])

    # Filters cores
    for i in range(10, 120, 10):
        mask, low_core = apply_lowpass(magnitude_spectrum, i)
        mask, high_core = apply_highpass(magnitude_spectrum, i)
        mask, pass_core = apply_bandpass(magnitude_spectrum, 120, i)
        plot_(low_core, high_core, pass_core, ['núcleo do filtro passa-baixa', 'núcleo do filtro passa-alta', 'núcleo do filtro passa-faixa', 'núcleo dos filtros - raio {}'.format(i)])

        mask, low_core = apply_lowpass(fshift_before, i)
        mask, high_core = apply_highpass(fshift_before, i)
        mask, pass_core = apply_bandpass(fshift_before, 120, i)

        low_ifft = ifft_ifftshift(low_core)
        high_ifft = ifft_ifftshift(high_core)
        pass_ifft = ifft_ifftshift(pass_core)
        
        plot_(low_ifft, high_ifft, pass_ifft, ['imagem após filtragem passa-baixa', 'imagem após filtragem passa-alta', 'imagem após filtragem passa-faixa', 'resultados dos filtros aplicados - raio {}'.format(i)])

    # Compression with kmeans
    kmeans = []
    k = [8, 16, 32, 64, 128, 256] #bits
    for comp in k:
        comp_img = compression_kmeans(img, comp)
        kmeans.append(comp_img)

    plot_6(kmeans, k, 'kmeans compression')

        
    # Compression in frequency
    img_comp, title = compression_(img)
    plot_6(img_comp, title, 'frequency compression')

    #rotation 45º
    dst = rotation_45(img)
    fshift_dst, magnitude_dst = fft_fftshift(dst)

    plt.subplot(121),plt.imshow(dst, cmap = 'gray'),plt.title('Rotation 45º')
    plt.subplot(122),plt.imshow(magnitude_dst, cmap = 'gray'),plt.title('Magnitude Spectrum')
    plt.show()