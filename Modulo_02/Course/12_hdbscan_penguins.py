from sklearn.metrics.cluster import silhouette_score
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import hdbscan
from sklearn.preprocessing import StandardScaler

df = sns.load_dataset("penguins")

variables = [
    "bill_length_mm",
    "bill_depth_mm",
    "flipper_length_mm",
    "body_mass_g"
]

data = df.dropna(subset=variables).copy()

x = data[variables]

scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)

model = hdbscan.HDBSCAN(metric='euclidean', min_cluster_size=10, min_samples=5)

labels = model.fit_predict(x_scaled)

data['cluster'] = labels

print(f"Cantidad de clusters\n{data['cluster'].value_counts()}")

clusters = set(labels) - {-1}

print(f"Número de clusters: {len(clusters)}")
print(f"Pingüinos considerados ruido: {sum(labels == -1)}")

mask = labels != -1
valid_labels = labels[mask]
x_valid = x_scaled[valid_labels]

group_quantity = len(np.unique(valid_labels))

if 1 <= group_quantity <= len(valid_labels):
    silhouette = silhouette_score(x_valid, valid_labels)
    print(f"Silhouette score: {silhouette}")
else:
    print(f"Silhouette can not be calculated")

sns.scatterplot(data=data, x="bill_length_mm", y="bill_depth_mm", hue="cluster", palette="pastel", alpha=0.7)
plt.title("Clusters con  pingüinos usando HDBSCAN")
plt.xlabel("bill_length_mm")
plt.ylabel("bill_depth_mm")
plt.tight_layout()
plt.show()
