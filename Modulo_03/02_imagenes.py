import numpy as np
from PIL import Image
from matplotlib import pyplot as plt

image = Image.open("./imagen.jpeg")

matrix_rgb = np.array(image)
print(f"Primer pixel: {matrix_rgb[0, 0]} | Shape: {matrix_rgb.shape}")

# Convertir a grayscale
image_gray = image.convert("L")
matrix_gray = np.array(image_gray)
print(f"Primer pixel: {matrix_gray[0, 0]} | Shape: {matrix_gray.shape}")

# Convertir a blanco y negro
umbral = 128
matrix_bw = np.where(matrix_gray >= umbral, 255, 0).astype(np.uint8)

plt.subplot(1, 3, 1)
plt.imshow(matrix_rgb)
plt.title("Imagen original")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(matrix_gray, cmap="gray", vmin=0, vmax=255)
plt.title("Imagen Grayscale")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(matrix_bw, cmap="gray")
plt.title("Imagen en blanco y negro")
plt.axis("off")

plt.tight_layout()
plt.show()