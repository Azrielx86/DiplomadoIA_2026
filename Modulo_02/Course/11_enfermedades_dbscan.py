from typing import Any
from sklearn.metrics.pairwise import pairwise_distances
from typing import List
from sklearn.metrics import silhouette_score
from sklearn.cluster import DBSCAN

import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
n_samples = 200

sintomas: List[str] = ["fiebre", "tos", "dolor_cabeza", "nauseas", "diarrea", "fatiga", "erupciones"]

probabilidades: List[float] = [0.55, 0.45, 0.60, 0.35, 0.30, 0.65, 0.20]

datos = rng.binomial(n=1, p=probabilidades, size=(n_samples, len(sintomas)))

df = pd.DataFrame(datos, columns=sintomas)
x = df[sintomas].to_numpy(dtype=bool)
distances = pairwise_distances(x, metric="jaccard")

model = DBSCAN(eps=0.35)
labels = model.fit_predict(distances)

df["Cluster"] = labels

cluster_num = len(np.unique(labels))
print(f"Clusters: {cluster_num}")

print(df["Cluster"].value_counts())

resumen: list[Any] = []

for cluster, group in df.groupby("Cluster"):
    frequencies = group[sintomas].mean()
    sintomas_cluster = [sintoma.replace('_', ' ').capitalize() for sintoma in sintomas if frequencies[sintoma] >= 0.7]

    if sintomas_cluster:
        sintomas_text = ", ".join(sintomas_cluster)
    else:
        sintomas_text = "Sin síntomas predominantes"

    resumen.append({
        "cluster": cluster,
        "infectados": len(group),
        "sintomas": sintomas_text,
    })

tabla_resumen = pd.DataFrame(resumen, columns=["cluster", "infectados", "sintomas"])
print(tabla_resumen)

label_filter = labels != -1
valid_labels = labels[label_filter]
cantidad_grupos = len(np.unique(valid_labels))

valid_distances = distances[np.ix_(label_filter, label_filter)]

if cantidad_grupos >= 2:
    silueta = silhouette_score(distances, labels)
    print(f"Silhouette score: {silueta}")
