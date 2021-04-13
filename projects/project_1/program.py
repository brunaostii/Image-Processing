import numpy as np
import argparse
import cv2 
import matplotlib.pyplot as plt

h1 = np.array([[-1,0,1], [-2,0,2], [-1,0,1]])
h2 = np.array([[-1,-2,-1], [0,0,0], [1,2,1]])
h3 = np.array([[-1,-1,-1], [-1,8,-1], [-1,-1,-1]])
h4 = 1/9* np.ones((3,3), dtype='uint8')
h5 = np.array([[-1,-1,2], [-1,2,-1], [2,-1,-1]])
h6 = np.array([[2,-1,-1], [-1,2,-1], [-1,-1,2]])
h7 = np.array([[0,0,1], [0,0,0], [-1,0,0]])
h8 = np.array([[0,0,-1,0,0], [0,-1,-2,-1,0], [-1,-2,16,-2,-1], [0,-1,-2,-1,0], [0,0,-1,0,0]])
h9 = 1/256 * np.array([[1,4,6,4,1], [4,16,24,16,4], [6,24,36,24,6], [4,16,24,16,4], [1,4,6,4,1]])

def plot_filters(fil_1,fil_2,fil_3,fil_4,fil_5,fil_6,fil_7,fil_8,fil_9):
    '''Essa função faz a exibição do resultado de todos os filtros para comparação '''

    plt.figure(figsize=(20,20))

    plt.subplot(3,3,1)
    plt.axis("off")
    plt.title('H1 - Derivada parcial em x')
    plt.imshow(fil_1, cmap='gray', vmin=0, vmax=255)

    plt.subplot(3,3,2)
    plt.axis("off")
    plt.title('H2 - Derivada parcial em y')
    plt.imshow(fil_2, cmap='gray', vmin=0, vmax=255)

    plt.subplot(3,3,3)
    plt.axis("off")
    plt.title('H3 - Laplaciano')
    plt.imshow(fil_3, cmap='gray', vmin=0, vmax=255)

    plt.subplot(3,3,4)
    plt.axis("off")
    plt.title('H4 - Média')
    plt.imshow(fil_4, cmap='gray', vmin=0, vmax=255)

    plt.subplot(3,3,5)
    plt.axis("off")
    plt.title('H5')
    plt.imshow(fil_5, cmap='gray', vmin=0, vmax=255)

    plt.subplot(3,3,6)
    plt.axis("off")
    plt.title('H6')
    plt.imshow(fil_6, cmap='gray', vmin=0, vmax=255)

    plt.subplot(3,3,7)
    plt.axis("off")
    plt.title('H7')
    plt.imshow(fil_7, cmap='gray', vmin=0, vmax=255)

    plt.subplot(3,3,8)
    plt.axis("off")
    plt.title('H8')
    plt.imshow(fil_8, cmap='gray', vmin=0, vmax=255)

    plt.subplot(3,3,9)
    plt.axis("off")
    plt.title('H9 - Gaussiano')
    plt.imshow(fil_9, cmap='gray', vmin=0, vmax=255)

    plt.show()


def plot_colour(img, img_title):
    '''Essa função exibe imagens coloridas usando matplotlib.pyplot'''

    plt.axis("off")
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title(img_title)
    plt.show()


def split_channels(img):
    '''Essa função retorna os canais da imagem separados'''
    return img[:, :, 2], img[:, :, 1], img[:, :, 0]

def plot_rgb(r,g,b):
    '''Essa função exibe cada canal da imagem separado na foto'''

    plt.figure(figsize=(10,15))

    plt.subplot(1,3,1)
    plt.axis("off")
    plt.title('Red')
    plt.imshow(r, cmap='Reds')

    plt.subplot(1,3,2)
    plt.axis("off")
    plt.title('Green')
    plt.imshow(g, cmap='Greens')

    plt.subplot(1,3,3)
    plt.axis("off")
    plt.title('Blue')
    plt.imshow(b, cmap='Blues')

    plt.show()

def plot_mono(img, img_title):
    '''Essa função exibe imagens monocromáticas'''

    plt.imshow(img, cmap='gray', vmin=0, vmax=255)
    plt.axis('off')
    plt.title(img_title)
    plt.show()

def ex_1_1_a(img):
    '''Essa função separa os canais da imagem, aplica uma operação entre o canais, 
    limita a intensidade em 255, exibe a imagem de acordo com cada canal e a imagem final após as operações'''

    r, g, b = split_channels(img)
    
    img_rec = np.zeros(img.shape, dtype='uint8')
    img_rec[:,:,2] = 0.393*r + 0.769*g + 0.189*b
    img_rec[:,:,1] = 0.349*r + 0.686*g+ 0.168*b
    img_rec[:,:,0] = 0.272*r + 0.534*g+ 0.131*b

    img_rec[img_rec > 255] = 255
    plot_rgb(img_rec[:,:,2] ,img_rec[:,:,1], img_rec[:,:,0])
    plot_colour(img_rec, 'Saída processada')
    
    return img_rec

def ex_1_1_b(img):
    '''Essa função aplica operações e faz com que a imagem fique apenas com um canal visto que a operação faz com que a
    imagem tenha apenas uma dimensão, exibe a imagem final após as operações'''

    r, g, b = split_channels(img)
    img =  0.2989*r+ 0.5870*g+ 0.1140*b

    plt.imshow(img, cmap='gray', vmin=0, vmax=255)
    plt.axis('off')
    plt.title('Imagem de saída com apenas uma banda')
    plt.show()

    return img

def filter_apply(img, filter_win):
    '''Essa função é uma operação de convolução de uma matriz por uma janela utilizando operações vetoriais '''

    # Subdividir a imagem de entrada em subimagens com o mesmo tamanho do filtro
    sub_matrices =  np.lib.stride_tricks.sliding_window_view(img,  filter_win.shape)
    
    # Multiplica os valores do filtro por cada submatriz e retorna a soma de cada submatriz
    return np.einsum('ij,klij->kl', filter_win, sub_matrices)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--i', type=str, help='input image - must be in format png', required=True)
    parser.add_argument('--o', type=str, help='output image - must be in format png', required=False, default='output.png')
    parser.add_argument('--f', type=str, help = 'part of exercise to be executed \n RGB: \n \
                                            "1.1a" - Operation in RGB; "1.1b" - Operation in one band; \
                                            "1.2_h[1,2,3,4,5,6,7,8,9]; 1.2_sobel; 1.2_all" - Filters' , required=True)
    args = parser.parse_args()

    assert args.i.lower().endswith(('.png')), 'The input format must be png!'
    assert args.o.lower().endswith(('.png')), 'The input format must be png!'
    

    if args.f == '1.1a':
        img = cv2.imread(args.i)
        print("Tipo de entrada: {}, Formato: {}".format(type(img), img.shape))

        img = ex_1_1_a(img)
        print("Tipo de saída: {}, Formato: {}".format(type(img), img.shape))
        _ = cv2.imwrite(args.o, img)
        
    elif args.f == '1.1b':
        print('\n'+ args.f +'...')
        img = cv2.imread(args.i)
        print("Tipo de entrada: {}, Formato: {}".format(type(img), img.shape))

        img = ex_1_1_b(img)
        print("Tipo de saída: {}, Formato: {}".format(type(img), img.shape))
        _ = cv2.imwrite(args.o, img)

    elif '1.2_h' in args.f:
        print(args.f+'...')
        img = cv2.imread(args.i, cv2.IMREAD_GRAYSCALE)
        print("Tipo de entrada: {}, Formato: {}\n".format(type(img), img.shape))
        filter_h = args.f.replace('1.2_h', '')

        if filter_h == '1':
            img = filter_apply(img, h1)
            _ = cv2.imwrite(args.o, img)
            plot_mono(img, 'H1')
            
        elif filter_h == '2':
            img = filter_apply(img, h2)
            _ = cv2.imwrite(args.o, img)
            plot_mono(img, 'H2')

        elif filter_h == '3':
            img = filter_apply(img, h3)
            _ = cv2.imwrite(args.o, img)
            plot_mono(img, 'H3')

        elif filter_h == '4':
            img = filter_apply(img, h4)
            _ = cv2.imwrite(args.o, img)
            plot_mono(img, 'H4')

        elif filter_h == '5':
            img = filter_apply(img, h5)
            _ = cv2.imwrite(args.o, img)
            plot_mono(img, 'H5')

        elif filter_h == '6':
            img = filter_apply(img, h6)
            _ = cv2.imwrite(args.o, img)
            plot_mono(img, 'H6')

        elif filter_h == '7':
            img = filter_apply(img, h7)
            _ = cv2.imwrite(args.o, img)
            plot_mono(img, 'H7')

        elif filter_h == '8':
            img = filter_apply(img, h8)
            _ = cv2.imwrite(args.o, img)
            plot_mono(img, 'H8')

        elif filter_h == '9':
            img = filter_apply(img, h9)
            _ = cv2.imwrite(args.o, img)
            plot_mono(img, 'H9')
        
    elif args.f == '1.2_sobel':
        print('Sobel...')
        img = cv2.imread(args.i, cv2.IMREAD_GRAYSCALE)
        filter_h1 = filter_apply(img, h1)
        filter_h2 = filter_apply(img, h2)
        filter_h1_h2 =  np.sqrt(filter_h1**2 + filter_h2**2)
        print("Tipo de saída: {}, Formato: {}".format(type(img), img.shape))
        _ = cv2.imwrite(args.o, filter_h1_h2)

        plot_mono(filter_h1_h2 , 'Filtro de Detecção de Bordas - Sobel')

    elif args.f == '1.2_all':
        print('Todos...')
        img = cv2.imread(args.i, cv2.IMREAD_GRAYSCALE)
        print("Tipo de entrada: {}, Formato: {}\n".format(type(img), img.shape))
        fil_1 = filter_apply(img, h1)
        fil_2 = filter_apply(img, h2)
        fil_3 = filter_apply(img, h3)
        fil_4 = filter_apply(img, h4)
        fil_5 = filter_apply(img, h5)
        fil_6 = filter_apply(img, h6)
        fil_7 = filter_apply(img, h7)
        fil_8 = filter_apply(img, h8)
        fil_9 = filter_apply(img, h9)

        plot_filters(fil_1,fil_2,fil_3,fil_4,fil_5,fil_6,fil_7,fil_8,fil_9)
