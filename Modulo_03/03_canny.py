from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import cv2

image = cv2.imread("imagen.jpeg", cv2.IMREAD_GRAYSCALE)

borders = cv2.Canny(image, 100, 200)

plt.figure(figsize=(10, 6))
plt.subplot(1, 2, 1)
plt.imshow(image, cmap="gray", vmin=0, vmax=255)  # cmap de tonalidades grises, minimo 0, max 255
plt.title("Imagen original en\nescala de grises")
plt.axis("off")  # Para que no salgan los valores de ejes X y Y

plt.subplot(1, 2, 2)
plt.imshow(borders, cmap="gray")
plt.title("Imagen con detección\nde bordes aplicando Canny")
plt.axis("off")  # Para que no salgan los valores de ejes X y Y

plt.tight_layout()  # Para que se ajuste al frame o ventana
plt.show()
