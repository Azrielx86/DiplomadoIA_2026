# Algoritmo Random Forest Isolation (para casos atípicos)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, confusion_matrix, classification_report)

df = pd.read_csv("../irisData.csv")
print(df.shape)

# Seleccionar dos especies de las flores
df = df[df["Species"].isin(["Iris-versicolor", "Iris-virginica"])].copy()

df["Clase"] = (df["Species"] == "Iris-virginica").astype(int)

# Partir el dataset en "X" y "y"
X = df[["PetalLengthCm"]]
y = df["Clase"]

# Separar en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# ---------- Parte Importante --------------------


# generación del modelo
modelo = LogisticRegression(solver="lbfgs", max_iter=100000, n_jobs=-1)

# Pasamos los datos de entrenamiento para el fit
modelo.fit(X_train, y_train)

y_pred = modelo.predict(X_test)

print("Resultados")
print("Accuracy", accuracy_score(y_test, y_pred))

print("Matriz de confusión")
print(confusion_matrix(y_test, y_pred))

print("Reporte")
print(classification_report(y_test, y_pred, target_names=["Iris-versicolor", "Iris-virginica"]))

# Obtener el intercepto (B0 de la regresión logística binaria)
b0 = modelo.intercept_[0]

# Obtener los valores de los coeficientes
b1 = modelo.coef_[0][0]

print("Modelo")
print(modelo)

print("Intercepto ", b0)
print("Coeficientes", b1)

linea = pd.DataFrame({"PetalLengthCm": np.linspace(X['PetalLengthCm'].min() - 0.5,
                                                   X['PetalLengthCm'].max() + 0.5,
                                                   300)})

probabilidades = modelo.predict_proba(linea)[:, 1]

versicolor = df[df["Clase"] == 0]
plt.scatter(versicolor["PetalLengthCm"], versicolor["Clase"], color="blue", alpha=0.6, label="Versicolor")

virginica = df[df["Clase"] == 1]
plt.scatter(virginica["PetalLengthCm"], virginica["Clase"], color="green", alpha=0.6, label="Virginica")

plt.plot(linea["PetalLengthCm"], probabilidades, color="red", label="Regresión Logística")

plt.axhline(y=0.5, color="gray", label="Umbral de separación de probabilidad")

plt.xlabel("Longitud del pétalo")
plt.ylabel("Probabilidad de flor virginica")

plt.legend()
plt.show()
