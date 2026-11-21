from pandas import DataFrame
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df: DataFrame = pd.read_csv("../irisData.csv")

print(df.head())
print(f"Shape: {df.shape}")

# Un histograma de la longitud de un pétalo
sns.histplot(data=df, x="SepalLengthCm", bins=10)
plt.title("Sepal Length Distribution")
plt.xlabel("Sepal Length")
plt.ylabel("Frequency")
plt.show()

# Un histograma de la longitud de un pétalo por especie
sns.histplot(data=df, x="SepalLengthCm", bins=20, hue="Species", alpha=0.5, palette="viridis")
plt.title("Sepal Length Distribution per Species")
plt.xlabel("Sepal Length")
plt.ylabel("Frequency")
plt.show()
