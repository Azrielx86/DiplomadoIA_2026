from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

imagen = Image.open("imagen.jpeg").convert("L")
matriz = np.array(imagen)

alto, ancho = matriz.shape  # renglones y columnas a alto y ancho

# 2. MÁSCARAS DE SOBEL

sobel_x = np.array([[-1, 0, 1], [-1, 0, 2], [-1, 0, 1]])  # Matriz de Prewitt en X

sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])  # Matriz de Prewitt en Y

bordes = np.zeros((alto, ancho))

for i in range(1, alto - 1):
    for j in range(1, ancho - 1):
        gx = 0
        gy = 0
        # Iterar en los filtros
        for k in range(3):
            for l in range(3):
                pixel = int(matriz[i + k - 1, j + l - 1])
                # 3. Cálculo del borde
                gx += pixel * sobel_x[k, l]  # Gx
                gy += pixel * sobel_y[k, l]  # Gy
        bordes[i, j] = np.sqrt(gx * 2 + gy * 2)  # G

if bordes.max() > 0:
    bordes = bordes / bordes.max() * 255

bordes = bordes.astype(np.uint8)

plt.figure(figsize=(10, 6))

plt.subplot(1, 2, 1)
plt.imshow(matriz, cmap="gray", vmin=0, vmax=255)  # cmap de tonalidades grises, minimo 0, max 255
plt.title("Imagen original en escala de grises")
plt.axis("off")  # Para que no salgan los valores de ejes X y Y

plt.subplot(1, 2, 2)
plt.imshow(bordes, cmap="gray", vmin=0, vmax=255)
plt.title("Imagen con detección de bordes aplicando Prewitt")
plt.axis("off")  # Para que no salgan los valores de ejes X y Y

plt.tight_layout()  # Para que se ajuste al frame o ventana
plt.show()
