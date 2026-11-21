from sklearn.metrics import silhouette_score
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


def interpret(value: float | int) -> str:
    if value > 0.40:
        return "HIGH"
    elif value < -0.40:
        return "LOW"
    else:
        return "MID "


df = pd.read_csv("../Wholesale.csv")

print(f"Shape: {df.shape}")

x = df[["Fresh", "Milk", "Grocery", "Frozen", "Detergents_Paper", "Delicassen"]]

scaler = StandardScaler()

x_scaled = scaler.fit_transform(x)

k_means = KMeans(n_clusters=5, random_state=0)

k_means.fit(x_scaled)

# Falta algo

centroids = scaler.inverse_transform(k_means.cluster_centers_)
centroids_df = pd.DataFrame(data=centroids, columns=["Centroid"])

print(centroids_df)

average = df[x.columns].mean()
cluster_resume = df.groupby('Cluster').mean(numeric_only=True)
relative_diff = (cluster_resume[x.columns] - average) / average
inference_table = relative_diff.map(interpret)

print(inference_table)

results = []

for k in range(2, 10):
    modelo = KMeans(n_clusters=k, random_state=42)
    labels = modelo.fit_predict(x_scaled)
    score = silhouette_score(x_scaled, labels)
    results.append((k, score))

results_df = pd.DataFrame(results, columns=["k", "silhouette_score"])

print("Resultados de K")
print(results_df)