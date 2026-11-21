import pandas as pd

df = pd.read_csv("../Data/data.csv")[:5]

df["edad"] = df["edad"].astype("Int8").sort_values(ascending=True)

nulls = df.isnull().sum()
print(nulls)

df = df.dropna()

print(" Aplicación de una función ")

df["promedio_round"] = df["promedio"].apply(round)
df["nombre"] = df["nombre"].apply(str.upper)

print(df.groupby("edad")["edad"])
