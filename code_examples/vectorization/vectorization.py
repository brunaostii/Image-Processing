import numpy as np

1. Contar o número de transições de Falso para Verdadeiro em uma sequência.

Exemplo: x = np.array([False, True, False, False, True])
Saída: 2

np.random.seed(123)
x = np.random.choice([False, True], size=10000)

# não vetorizada
def conta_transicoes(x):
    t = 0
    for i, j in zip(x[:-1], x[1:]):
        if (j and not i):
           t += 1
    return t

conta_transicoes(x)

# vetorizada
np.count_nonzero([x[:-1] < x[1:]])

2. Selecionar apenas números ímpares e eleva-os ao quadrado.

Exemplo: x = np.array([1, 2, 3, 4])
Saída: [1, 9]

print(x[x % 2 == 1] ** 2)

3. Somar todos os elementos divisíveis por 5.

Exemplo: x = np.array([1, 5, 12, 15, 20, 22])
Saída: 40

print(x[x % 5 == 0].sum())

4. Somar cada n valores em um vetor.

Exemplo: x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]) e n = 4
Saída: [10, 26, 42]

# número de elementos não precisa ser múltiplo de n
print(np.add.reduceat(x, np.arange(0, len(x), n)))

# número de elementos precisa ser múltiplo de n
print(np.reshape(x, (-1, n)).sum(axis=1))

# observação: reshape(-1, c): significa que o número de linhas é desconhecido (número de colunas = c)
# também pode ser: reshape(r, -1): r linhas, com número de colunas desconhecido

5. Trocar todos os valores negativos em um vetor por zeros.

Exemplo: x = np.array([-3, -2, -1, 0, 1, 2, 3])
Saída: [0, 0, 0, 0, 1, 2, 3]

# primeira forma vetorizada
x[x < 0] = 0

# segunda forma vetorizada
x = np.maximum(x, 0)

# terceira forma vetorizada
x = np.where(x < 0, 0, x) 

6. Contar número de valores pares em um vetor.

Exemplo: x = np.array([1, 2, 3, 4, 5, 6, 7])
Saída: 3

np.sum(x % 2 == 0)

7. Encontrar máximos (picos) locais.

Exemplo: x = np.array([1, 2, 3, 4, 5, 4, 3, 2, 1, 2, 3, 2, 1, 2, 3, 4, 5, 6, 5, 4, 3, 2, 1])
Saída: [5, 3, 6]

picos = x[1:-1][np.diff(np.diff(x)) < 0]

8. Verifica se número é primo.

# n: não negativo
def primo(n):
    if (n % 2 == 0 and n > 2):
       return False
    return all(n % i for i in range(3, int(np.sqrt(n)) + 1, 2))

primo(7)
primo(8)

9. Construir o histograma de uma imagem.

import numpy as np
import cv2
from matplotlib import pyplot as plt

# read image
img = cv2.imread("baboon.png")
row, col, bands = img.shape

# create histogram (non-vectorized version)
hist = np.zeros((256), np.uint64)
for i in range(0, row):
    for j in range(0, col):
        hist[img[i,j]] += 1

# show histogram
x = np.arange(0, 256)
plt.bar(x, hist, color="gray")
plt.show()

# create histogram (vectorized version)
hist = np.bincount(img.ravel(), minlength=256)

# create histogram (vectorized version)
hist = (img.ravel() == np.arange(256)[:,None]).sum(axis=1)

# create histogram (vectorized version)
plt.hist(img.flatten(), bins=256, range=(0, 255))
plt.show()

# create histogram (vectorized version)
hist, bin_edges = np.histogram(img[:, :, 0], bins=256, range=(0, 255))
plt.plot(bin_edges[0: -1], hist, color="g")
plt.show()

# create histogram (vectorized version)
hist = cv2.calcHist([img], [0], None, [256], [0, 255]) 
plt.plot(hist) 
plt.show()

