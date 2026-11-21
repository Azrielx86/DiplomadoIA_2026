import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, fcluster
from seaborn.matrix import dendrogram
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("../irisData.csv")

columns = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']

scaler = StandardScaler()

x = scaler.fit_transform(df[columns])
z = linkage(x, method="single")

plt.figure(figsize=(15, 8))

dendrogram(z)
plt.title("Dendrogram de Iris NS")
plt.xlabel("Flores")
plt.ylabel("Distancia que fusiona las flores")

plt.show()

ks = range(1, 11)
siluetas = []
for k in ks:
    etiquetas = fcluster(z, t=k, criterion="distance")
    score = silhouette_score(x, etiquetas)
    siluetas.append(score)

mejor_k = ks[np.argmax(siluetas)]

plt.plot(list(ks), siluetas, marker="o")
plt.axvline(x=mejor_k, color="red", linestyle="--", label="Mejor k")
plt.title("Gráfica silhouette_score")
plt.xlabel("Valores de K")
plt.ylabel("Valores de la silueta")
plt.legend()
