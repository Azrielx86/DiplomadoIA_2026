from sklearn.metrics.cluster import silhouette_score
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import hdbscan
from sklearn.preprocessing import StandardScaler

iris = sns.load_dataset("iris")
iris.info()

x = iris.drop(columns=['species'])

scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)

model = hdbscan.HDBSCAN(metric='euclidean', min_cluster_size=5, min_samples=3)

labels = model.fit_predict(x_scaled)

iris['cluster'] = labels

print(f"Cantidad de clusters\n{iris['cluster'].value_counts()}")

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

sns.scatterplot(data=iris, x="petal_length", y="petal_width", hue="cluster", palette="pastel", alpha=0.7)
plt.title("Clusters con pingüinos usando HDBSCAN")
plt.xlabel("petal_length")
plt.ylabel("petal_width")
plt.tight_layout()
plt.show()

