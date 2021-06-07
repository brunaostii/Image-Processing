from cv2 import cv2
import matplotlib.pyplot as plt 
import numpy as np
import argparse


def plot_input_gray(img, gray, out):
    plt.figure(figsize=(10,10))
    plt.subplot(121)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title('Imagem RGB')
    plt.subplot(122)
    plt.imshow(gray,cmap='gray', vmin=0, vmax=255)
    plt.title('Imagem monocromática')
    plt.savefig(out +'_input_color_gray.png')
    plt.show()

def plot_binary_img(all_contours, out):
    plt.imshow((all_contours * 255).astype(np.uint8), cmap='gray')
    plt.xlabel("contorno dos objetos")
    plt.savefig(out + '_binary_contour.png')
    plt.show()

def plot_labels(img, out):
    plt.figure(figsize=(5,5))
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.xlabel('regiões rotuladas')
    plt.savefig(out + 'labeled_regions.png')
    plt.show()

def eccentricity_(moments):
    a1 = (moments['mu20'] + moments['mu02']) / 2
    a2 = np.sqrt(4 * moments['mu11'] ** 2 + (moments['mu20'] -moments['mu02']) ** 2) / 2
    ecc = np.sqrt(1 - (a1 - a2) / (a1 + a2))
    return ecc

def extract_props(contours, img):
    aux = 0
    areas = []
    print("Extração da propriedade dos objetos\n")
    for contour in contours[1:]:
        
        M = cv2.moments(contour) #calculo do momento
        
        cX = int(M["m10"] / M["m00"]) #calculo da coordenada x central
        cY = int(M["m01"] / M["m00"]) #calculo da coordenada y central
        area = cv2.contourArea(contour) #calculo da área de contorno
        perimeter = cv2.arcLength(contour,True) #calculo do perímetro
        
        
        hull = cv2.convexHull(contour) #Área do fecho convexo
        hull_area = cv2.contourArea(hull) #Área do contorno
        solidity = float(area)/hull_area  #Solidez
        areas.append(area)

        eccentricity = eccentricity_(M)
        
    
        print('região {}:  centróide: {}  perímetro: {:,.6f} área: {:,.0f} excentricidade: {:,.6f} solidez: {:,.6f} '.format(aux, (cX, cY), perimeter, area, eccentricity, solidity))
        
    #     Descomentar se gostar das figuras + borda realçada 
    #     img = cv2.drawContours(img, contours[1:], -1, (0,255,0), 3)
        img= cv2.putText(img, str(aux), (cX, cY), cv2.FONT_HERSHEY_SIMPLEX, 0.25, (0, 0, 0), 1) #adicionando labels ao centro de cada imagem, com tamanho de 0,25, fonte: FONT_HERSHEY_SIMPLEX e 1 px de largura.
        
        aux +=1
    
    return img, areas

def histograma_area(areas, out):
    areas = np.array(areas)
    small = areas[areas < 1500]
    medium = areas[(areas < 3000) & (areas >= 1500)]
    big = areas[areas >= 3000]

    print("Histograma da área dos objetos\n")
    print("número de regiões pequenas: {} \nnúmero de regiões médias: {} \nnúmero de regiões grandes: {}".format(len(small), len(medium), len(big)))


    values = [len(small), len(medium), len(big)]
    labels = ['areas < 1500', '1500 <= areas < 3000', 'areas >= 3000']
    plt.figure(figsize = (6,4))
    plt.bar(labels, values)
    plt.xlabel("Área")
    plt.ylabel("Número de Objetos")
    plt.savefig(out + '_area_histogram.png')
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Project 3 - Measures')
    parser.add_argument('--i', type=str, help='image input', required=True)
    parser.add_argument('--o', type=str, help='name_output', required=False, default='image')
    args = parser.parse_args()

    img = cv2.imread(args.i)
    out = args.o
    # Transformação de cores
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    plot_input_gray(img, gray, out)
    
    # Escala de cinza -> binária
    ret, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)

    # Contorno dos objetos
    contours, hierarquia = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    white_back = np.ones((img.shape[0], img.shape[1]))
    all_contours = cv2.drawContours(white_back, contours[1:], -1, (0,255,0), 2)
    plot_binary_img(all_contours, out)

    #Extração de propriedades do objeto
    img_labels, areas = extract_props(contours, img)
    plot_labels(img_labels, out)

    # Histograma da área dos objetos
    print('\n')
    histograma_area(areas, out)