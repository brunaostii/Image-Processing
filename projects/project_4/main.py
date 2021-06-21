from cv2 import cv2
import matplotlib.pyplot as plt 
import numpy as np
from itertools import product
from scipy.spatial import distance
import pandas as pd
from skimage.feature import greycomatrix, greycoprops
import argparse
import ast
from skimage import io, color
from skimage.feature import local_binary_pattern

def lbp_histogram(img, p=8, r=1):
    '''P -> Number of circularly symmetric neighbour set points
        R -> Radius of circle (space resolution of the operator)'''
    patterns = local_binary_pattern(img, p, r) 
    n_bins = int(patterns.max() + 1) 
    hist, _ = np.histogram(patterns, bins=n_bins, density=True, range=(0, n_bins))
    return hist, patterns

def append_compar(texture_1, tex):
    return {'texture': tex, 'entropy_0':texture_1[0][0][0], 'entropy_45':texture_1[0][0][1],\
        'entropy_90':texture_1[0][0][2], 'entropy_135':texture_1[0][0][3], 'contrast_0':texture_1[1][0][0],\
        'contrast_45':texture_1[1][0][1], 'contrast_90':texture_1[1][0][2], 'contrast_135':texture_1[1][0][3], \
        'angular_moment_0':texture_1[2][0][0], 'angular_moment_45':texture_1[2][0][1],\
        'angular_moment_90':texture_1[2][0][2], 'angular_moment_135':texture_1[2][0][3]}

def plot_image_grid(images, n_texture, output_name, ncols=None, cmap='gray' ):
    '''Plot a grid of images'''
    if not ncols:
        factors = [i for i in range(1, len(images)+1) if len(images) % i == 0]
        ncols = factors[len(factors) // 2] if len(factors) else len(images) // 4 + 1
    nrows = int(len(images) / ncols) + int(len(images) % ncols)
    imgs = [images[i] if len(images) > i else None for i in range(nrows * ncols)]
    f, axes = plt.subplots(nrows, ncols, figsize=(3*ncols, 2*nrows))
    axes = axes.flatten()[:len(imgs)]
    aux = 0
    for img, ax in zip(imgs, axes.flatten()): 
        aux += 1
        if np.any(img):
            if len(img.shape) > 2 and img.shape[2] == 1:
                img = img.squeeze()
            ax.set_title(n_texture + " " + str(aux))
            ax.axis('off')
            ax.imshow(img, cmap=cmap)
    plt.savefig(output_name + "_lbp_gray.png")
    plt.show()

def plot_4_rgb(text_1, text_2, text_3, text_4, output_name):
    plt.figure(figsize=(15,15))

    plt.subplot(221)
    plt.axis('off')
    plt.title("Textura 1")
    plt.imshow(cv2.cvtColor(text_1, cv2.COLOR_BGR2RGB))
   
    plt.subplot(222)
    plt.title("Texture 2")
    plt.imshow(cv2.cvtColor(text_2, cv2.COLOR_BGR2RGB))
    plt.axis('off')
   
    plt.subplot(223)
    plt.title("Texture 3")
    plt.imshow(cv2.cvtColor(text_3, cv2.COLOR_BGR2RGB))
    plt.axis('off')

    plt.subplot(224)
    plt.title("Texture 4")
    plt.axis('off')
    plt.imshow(cv2.cvtColor(text_4, cv2.COLOR_BGR2RGB))
    plt.suptitle("Input Textures")
    plt.savefig(output_name + "_inputdata_.png")
    plt.show()

def comparation_of_textures(text_5, text_6, text_7, text_8, gray_5, gray_6, gray_7, gray_8, output_name):
    comparation_euclidean = pd.DataFrame(columns=['p', 'r', 'euclidean_distance_1_2', 'euclidean_distance_1_3', 'euclidean_distance_1_4', \
                                       'euclidean_distance_2_3', "euclidean_distance_2_4", "euclidean_distance_3_4"])

    p = [4,8,16]
    r = np.arange(1,17,1)
    
    hmax = max([hist_1.max(), hist_2.max(), hist_3.max(), hist_4.max()])
    fig, ax = plt.subplots(1, 4, figsize=(20,10))
    ax[0].imshow(cv2.cvtColor(text_5, cv2.COLOR_BGR2RGB))
    ax[0].axis('off')
    ax[0].set_title('Texture 1')
    ax[1].imshow(cv2.cvtColor(text_6, cv2.COLOR_BGR2RGB))
    ax[1].axis('off')
    ax[1].set_title('Texture 2')
    ax[2].imshow(cv2.cvtColor(text_7, cv2.COLOR_BGR2RGB))
    ax[2].axis('off')
    ax[2].set_title('Texture 3')
    ax[3].imshow(cv2.cvtColor(text_8, cv2.COLOR_BGR2RGB))
    ax[3].axis('off')
    ax[3].set_title('Texture 4')
    plt.savefig(output_name + "_input_lbp_.png")
    plt.show()
    
    for p,r in product(p, r):
        
        print("Number of points p: {}, radius: {}".format(p, r))
        
        hist_5, patterns_5 = lbp_histogram(gray_5, p, r)
        hist_6, patterns_6 = lbp_histogram(gray_6, p, r)
        hist_7, patterns_7 = lbp_histogram(gray_7, p, r)
        hist_8, patterns_8 = lbp_histogram(gray_8, p, r)
        
        euclidean_distance_1_2 = distance.euclidean(hist_5, hist_6)
        euclidean_distance_1_3 = distance.euclidean(hist_5, hist_7)
        euclidean_distance_1_4 = distance.euclidean(hist_5, hist_8)
        euclidean_distance_2_3 = distance.euclidean(hist_6, hist_7)
        euclidean_distance_2_4 = distance.euclidean(hist_6, hist_8)
        euclidean_distance_3_4 = distance.euclidean(hist_7, hist_8)
        
        print("Euclidean distance between Texture 1 e 2: {}". format(euclidean_distance_1_2))
        print("Euclidean distance between Texture 1 e 3: {}". format(euclidean_distance_1_3))
        print("Euclidean distance between Texture 1 e 4: {}". format(euclidean_distance_1_4))
        print("Euclidean distance between Texture 2 e 3: {}". format(euclidean_distance_2_3))
        print("Euclidean distance between Texture 2 e 4: {}". format(euclidean_distance_2_4))
        print("Euclidean distance between Texture 3 e 4: {}". format(euclidean_distance_3_4))

    
        comparation_euclidean = comparation_euclidean.append({'p': p, 'r': r, 'euclidean_distance_1_2': euclidean_distance_1_2, \
                                      'euclidean_distance_1_3': euclidean_distance_1_3, 'euclidean_distance_1_4': euclidean_distance_1_4, \
                                       'euclidean_distance_2_3':euclidean_distance_2_3, "euclidean_distance_2_4":euclidean_distance_2_4 ,\
                                        "euclidean_distance_3_4": euclidean_distance_3_4}, ignore_index=True)
        
        
        hmax = max([hist_1.max(), hist_2.max(), hist_3.max(), hist_4.max()])
        fig, ax = plt.subplots(1, 4, figsize=(20,5))
        ax[0].plot(hist_5)
        ax[0].set_ylim([0, hmax])
        ax[0].set_title('Texture 1')
        ax[0].axes.yaxis.set_ticklabels([])
        
        ax[1].plot(hist_6)
        ax[1].set_ylim([0, hmax])
        ax[1].set_title('Texture 2')
        ax[1].axes.yaxis.set_ticklabels([])

        
        ax[2].plot(hist_7)
        ax[2].set_ylim([0, hmax])
        ax[2].set_title('Texture 3')
        ax[2].axes.yaxis.set_ticklabels([])

       
        ax[3].plot(hist_8)
        ax[3].set_ylim([0, hmax])
        ax[3].set_title('Texture 4')
        ax[3].axes.yaxis.set_ticklabels([])
        plt.suptitle("LBP - Histograms (p = {}, r= {})".format(p, r))
        plt.savefig(output_name + "out_lbp_{}p_{}r.png".format(p, r))
        plt.show()
        print("----------------------------------------------------------------")
        
    return comparation_euclidean




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Textures - Project 4')
    parser.add_argument('-i', type=str, help='4 textures images in png', default="./images/textura5.png,images/textura6.png,./images/textura7.png,./images/textura8.png")
    parser.add_argument('-o', type=str, help="ouput name", default="output.png")
    args =  parser.parse_args()

    images = args.i.split(',')
    if len(images) < 4:
        print('Number of textures < 4')
        exit()

    for j in images:
        if ".png" not in j:
            print("Input not png image")
            exit()

    if ".png" not in args.o:
        print("ouput not png image")
        exit()

    output_name = args.o
    output_name = output_name.replace(".png", "")
    print(output_name)

    # Carregando as imagens
    print(images[0])
    text_1 = cv2.imread(images[0]) 
    text_2 = cv2.imread(images[1])
    text_3 = cv2.imread(images[2])
    text_4 = cv2.imread(images[3])
    
    # Transformando as imagens para tons de cinza
    gray_1 = cv2.cvtColor(text_1, cv2.COLOR_BGR2GRAY)
    gray_2 = cv2.cvtColor(text_2, cv2.COLOR_BGR2GRAY)
    gray_3 = cv2.cvtColor(text_3, cv2.COLOR_BGR2GRAY)
    gray_4 = cv2.cvtColor(text_4, cv2.COLOR_BGR2GRAY)

    plot_4_rgb(text_1, text_2, text_3, text_4, output_name)

    hist_1, patterns_1 = lbp_histogram(gray_1)
    hist_2, patterns_2 = lbp_histogram(gray_2)
    hist_3, patterns_3 = lbp_histogram(gray_3)
    hist_4, patterns_4 = lbp_histogram(gray_4)

    plot_image_grid([patterns_1, patterns_2, patterns_3, patterns_4], "Texture", output_name, ncols=None, cmap='gray')

    comparation_euclidean = comparation_of_textures(text_1, text_2, text_3, text_4, gray_1, gray_2, gray_3, gray_4, output_name)
    comparation_euclidean.to_csv("euclidean_distance_lbp.csv", index=False)

    comparation_greycomatrix = pd.DataFrame(columns=['entropy_0','entropy_45', 'entropy_90', 'entropy_135',\
                         'contrast_0', 'contrast_45', 'contrast_90', 'contrast_135', \
                         'angular_moment_0', 'angular_moment_45', 'angular_moment_90', \
                         'angular_moment_135'])


    texture_1 = [greycoprops(greycomatrix(gray_1, [1], [0 ,45, 90, 135]), prop=x) for x in ['dissimilarity', 'contrast', 'ASM']]
    texture_2 = [greycoprops(greycomatrix(gray_2, [1], [0 ,45, 90, 135]), prop=x) for x in ['dissimilarity', 'contrast', 'ASM']]
    texture_3 = [greycoprops(greycomatrix(gray_3, [1], [0 ,45, 90, 135]), prop=x) for x in ['dissimilarity', 'contrast', 'ASM']]
    texture_4 = [greycoprops(greycomatrix(gray_4, [1], [0 ,45, 90, 135]), prop=x) for x in ['dissimilarity', 'contrast', 'ASM']]

    comparation_greycomatrix = comparation_greycomatrix.append(append_compar(texture_1, 1), ignore_index=True)
    comparation_greycomatrix = comparation_greycomatrix.append(append_compar(texture_2, 2), ignore_index=True)
    comparation_greycomatrix = comparation_greycomatrix.append(append_compar(texture_3, 3), ignore_index=True)
    comparation_greycomatrix = comparation_greycomatrix.append(append_compar(texture_4, 4), ignore_index=True)

    print("Texture 1: ['entropy': {}, 'contrast':{}, 'angular moment': {}]\n".format(texture_1[0], texture_1[1], texture_1[2]))
    print("Texture 2: ['entropy': {}, 'contrast':{}, 'angular moment': {}]\n".format(texture_2[0], texture_2[1], texture_2[2]))
    print("Texture 3: ['entropy': {}, 'contrast':{}, 'angular moment': {}]\n".format(texture_3[0], texture_3[1], texture_3[2]))
    print("Texture 4: ['entropy': {}, 'contrast':{}, 'angular moment': {}]\n".format(texture_4[0], texture_4[1], texture_4[2]))

    comparation_greycomatrix.to_csv("coocurency_matrix.csv", index=False)