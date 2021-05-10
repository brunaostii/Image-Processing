from cv2 import cv2
import matplotlib.pyplot as plt
import argparse
from filters import *
from utils import *
from compression import *


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Project 2 - Frequency filtering')
    parser.add_argument('-i', type=str, help='image input', required=True)
    parser.add_argument('-f', type=str, help='c) colour, m) monochromatic', required=True)
    args = parser.parse_args()

    
    if args.f == 'c':
        print('Only monochromatic images!')
        exit()
    elif '.png' not in args.i:
        print('Please, insert an image png!!')
        exit() 

    img = cv2.imread(args.i, cv2.IMREAD_GRAYSCALE)
    
    origin_min = img.min()
    origin_max = img.max()


    # Original Image, Magnitude Spectrum and IFFT -> original image
    fshift_before, magnitude_spectrum = fft_fftshift(img)
    img_ifft = ifft_ifftshift(fshift_before, origin_min,origin_max)
    plot_(img, magnitude_spectrum, img_ifft, ['Imagem original ({} pixels)'.format(img.shape), 'espectro de Fourier (magnitude)', 'imagem após inversa de Fourier', 'imagem_original'])

    # Filters cores
    for i in range(10, 130, 10):
        mask, low_core = apply_lowpass(magnitude_spectrum, i)
        mask, high_core = apply_highpass(magnitude_spectrum, i)
        mask, pass_core = apply_bandpass(magnitude_spectrum, 120, i)
        
        mask, low_ = apply_lowpass(fshift_before, i)
        mask, high_=  apply_highpass(fshift_before, i)
        mask, pass_=  apply_bandpass(fshift_before, 120, i)

        low_ifft = ifft_ifftshift(low_,origin_min,origin_max)
        high_ifft = ifft_ifftshift(high_, origin_min, origin_max)
        pass_ifft = ifft_ifftshift(pass_, origin_min, origin_max)
        
        plot_6([low_core, high_core, pass_core, low_ifft, high_ifft, pass_ifft], \
            ['núcleo do filtro passa-baixa', 'núcleo do filtro passa-alta', 'núcleo do filtro passa-faixa','imagem após filtragem passa-baixa', 'imagem após filtragem passa-alta', 'imagem após filtragem passa-faixa' ], 'filtros_aplicados_raio_{}'.format(i))

        
    # Compression with kmeans
    kmeans = []
    k = [8, 16, 32, 64, 128, 256] #bits
    
    for comp in k:
        comp_img = compression_kmeans(img, comp)
        kmeans.append(comp_img)

    plot_6(kmeans, k, 'kmeans_compression')

        
    # Compression in frequency
    img_comp, title = compression_(img)
    plot_6(img_comp, title, 'Remocao_coeficientes_menores_limiar')

    #rotation 45º
    dst = rotation_45(img)
    fshift_dst, magnitude_dst = fft_fftshift(dst)

    plt.subplot(121),plt.imshow(dst, cmap = 'gray'),plt.title('Rotation 45º')
    plt.subplot(122),plt.imshow(magnitude_dst, cmap = 'gray'),plt.title('Magnitude Spectrum')
    plt.savefig('rotation45.png')
    plt.show()