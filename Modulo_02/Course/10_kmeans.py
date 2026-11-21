from matplotlib.axes import Axes
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
from typing import Literal
from typing import List

import matplotlib.pyplot as plt
import numpy as np
from numpy import random

import seaborn as sns
import pandas as pd
from sklearn.preprocessing import StandardScaler

SIZES: List[Literal["S", "M", "L"]] = ["S", "M", "L"]

np.random.seed(42)

# Definir monstruo: peligrosidad (1 - 10), fear_factor (1 - 10), Size (small, mid, large), local (true, false)

# Definir valor de peligrosidad

group_1 = np.random.normal(loc=[])
count = 1

dangerous: List[int] = []
fear: List[int] = []
sizes: List[Literal["S", "M", "L"]] = []
local: List[bool] = []

for i in range(200):
    p = random.uniform(1, 10)
    m = 0.7 * p + np.random.normal(2, 1.8)
    p = np.clip(p, 1, 10)
    m = np.clip(m, 1, 10)
    dangerous.append(p)
    fear.append(m)
    sizes.append(random.choice(["CH", "M", "G"]))
    local.append(random.choice([True, False]))

df = pd.DataFrame({"dangerous": dangerous, "fear": fear,
                   "size": sizes, "local": local})
print(df.head())

sns.scatterplot(data=df, x="dangerous", y="fear", color=
"gray", s=80, edgecolor="black", alpha=0.5)

plt.title("Distribución inicial de monstrous")
plt.xlabel("Peligrosidad")
plt.ylabel("Modulo")

plt.show()

scaler = StandardScaler()
x = df[["dangerous", "fear"]]

x_scaled = scaler.fit_transform(x)

k_values = list[int](range(2, 11))

inercias: list[float] = []
siluetas: list[float] = []
models: list[KMeans] = []

for k in k_values:
    model = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = model.fit_predict(x_scaled)
    models.append(model)
    inercias.append(model.inertia_)
    siluetas.append(silhouette_score(x_scaled, labels))

best_index = np.argmax(siluetas)
best_k = k_values[best_index]
kmeans = models[best_index]

print(f"Mejor número de clusters: {best_k}")
print(f"Mejor valor de silhouette_score: {siluetas[best_index]}")

# =============
axes: list[Axes]
fix, axes = plt.subplots(nrows=1, ncols=2)

# Gráfica 1
axes[0].plot(k_values, inercias, marker="D", color="orange")
axes[0].axvline(x=best_k, color="red", linestyle="--", label="Mejor valor de K")
axes[0].set_title("Método del codo")
axes[0].set_xlabel("Valores de K")
axes[0].set_ylabel("Inercia")
axes[0].legend()

# Gráfica 2
axes[1].plot(k_values, siluetas, marker="o", color="blue")
axes[1].axvline(x=best_k, color="red", linestyle="--", label="Mejor valor de K")
axes[1].set_title("Gráfica silhouette_score")
axes[1].set_xlabel("Valores de K")
axes[1].set_ylabel("Valores de la silueta")
axes[1].legend()

plt.tight_layout()
plt.show()

df["cluster"] = kmeans.labels_

sns.scatterplot(data=df, x="dangerous", y="fear", hue="cluster",
                s=80, palette="viridis", alpha=0.5,
                edgecolor="black")
plt.show()
