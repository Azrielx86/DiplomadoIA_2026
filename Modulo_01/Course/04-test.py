import numpy as np

datos = np.array([10, 12, 11, 10, 9, 50, 11, 10])
media = np.mean(datos)
std = np.std(datos)
anomalias = datos[np.abs(datos - media) > 2 * std]
print(anomalias)
