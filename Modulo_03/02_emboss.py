from PIL import Image, ImageFilter
import matplotlib.pyplot as plt

image = Image.open("./imagen.jpeg").convert("RGB")

image_emboss = image.filter(ImageFilter.EMBOSS)

plt.figure(figsize=(10, 6))
plt.subplot(1, 2, 1)
plt.imshow(image, cmap="gray", vmin=0, vmax=255)  # cmap de tonalidades grises, minimo 0, max 255
plt.title("Imagen original")
plt.axis("off")  # Para que no salgan los valores de ejes X y Y

plt.subplot(1, 2, 2)
plt.imshow(image_emboss, cmap="gray")
plt.title("Imagen con detección\nde bordes aplicando Emboss")
plt.axis("off")  # Para que no salgan los valores de ejes X y Y

plt.tight_layout()  # Para que se ajuste al frame o ventana
plt.show()
