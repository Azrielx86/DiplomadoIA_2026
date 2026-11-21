import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler

# Adelie = -1 clase negativa
adelie = np.array([
    [38, 185],
    [39, 188],
    [40, 190],
    [37, 182],
    [41, 192],
    [38, 187],
    [40, 185],
    [39, 191],
    [42, 194],
    [37, 186],
    [41, 189],
    [40, 193],
    [38, 190],
    [45, 207],
    [47, 211]
])

# Gentoo = +1 (clase positiva)
gentoo = np.array([
    [44, 200],
    [45, 205],
    [46, 210],
    [47, 212],
    [48, 215],
    [45, 208],
    [49, 218],
    [46, 205],
    [47, 210],
    [48, 212],
    [50, 220],
    [44, 204],
    [49, 214],
    [40, 190],
    [41, 192]
])

scaler = StandardScaler()

valor_C = 1
valor_coef = 1

X = np.vstack((adelie, gentoo))
y = np.array([-1] * 15 + [1] * 15)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = SVC(kernel='sigmoid', C=valor_C, gamma='scale', coef0=valor_coef)

model.fit(X_train, y_train)

score_train = model.score(X_train, y_train)
score_test = model.score(X_test, y_test)

y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

errors_train = np.sum(y_pred_train != y_train)
errors_test = np.sum(y_pred_test != y_test)

print(f"Train accuracy: {score_train * 100:.2f}%")
print(f"Test accuracy: {score_test * 100:.2f}%")

print(f"Train errors: {errors_train * 100:.2f}%")
print(f"Test errors: {errors_test * 100:.2f}%")

# f(x) = W * x + b
f = model.decision_function(X_train)

# shi = max(0, 1 - y * f(x))

shi = np.maximum(0, 1 - y_train * f)
print(f"=> Margin separator violations")

for i in range(len(X_train)):
    product = y_train[i] * f[i]

    if product > 1:
        state = "Producto mal clasificado"
    elif product == 1:
        state = "En la frontera"
    elif 0 < product < 1:
        state = "Dentro del márgen bien clasificado"
    else:
        state = "No viola el márgen"

    print(f"===> Pinguino: {i + 1} | Clase real: {y_train[i]} | f(x): {f[i]} | {state =}")

vectores = model.support_vectors_
alpha_y = model.dual_coef_

K = np.tanh(model._gamma * (vectores @ vectores.T) * valor_coef)
print("Valor de K")
# print(K)

# J = 1/2 ||W|| + C * sigma(shi)
termino_margen = 0.5 * (alpha_y @ K @ alpha_y.T)
termino_errores = valor_C * shi.sum()

J = termino_margen + termino_errores

print(f"J = 1/2 * {termino_margen} + {valor_C} * sigma({termino_errores})")

print(f"Vectores de soporte ({len(model.support_vectors_)}):")
print(model.support_vectors_)

print(classification_report(y_test, y_pred_test, labels=[-1, 1]))

matrix = confusion_matrix(y_test, y_pred_test, labels=[-1, 1])

print(matrix)

# Graficando el entorno del modelo

x_min = X[:, 0].min() - 3
x_max = X[:, 0].max() + 3

y_min = X[:, 1].min() - 3
y_max = X[:, 1].max() + 3

xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300), np.linspace(y_min, y_max, 300))

# Revertir escala
X_test = scaler.inverse_transform(X_test)
X_train = scaler.inverse_transform(X_train)
vectores = scaler.inverse_transform(vectores)

puntos = np.c_[xx.ravel(), yy.ravel()]
puntos_scaled = scaler.transform(puntos)

print(puntos)

Z = model.decision_function(puntos_scaled)
Z = Z.reshape(xx.shape)

plt.contourf(xx, yy, Z, levels=[-1000, 0, 1000], colors=["royalblue", "seagreen"], alpha=0.12)

# Pintar a los adelie (training)

plt.scatter(X_train[y_train == -1, 0], X_train[y_train == -1, 1], color="royalblue", s=70, label="Adelie (training)")
plt.scatter(X_train[y_train == 1, 0], X_train[y_train == 1, 1], color="seagreen", s=70, label="Gentoo (training)")

# Pintar a los de prueba
plt.scatter(X_test[y_test == -1, 0], X_test[y_test == -1, 1], color="royalblue", s=120, edgecolors="black", linewidths=2, label="Adelie (test)")
plt.scatter(X_test[y_test == 1, 0], X_test[y_test == 1, 1], color="seagreen", s=120, edgecolors="black", linewidths=2, label="Gentoo (test)")

plt.contour(xx, yy, Z, levels=[-1, 0, 1], colors=["black"], linestyles=['--', '-', '--'], linewidths=[1.5, 2.5, 1.5])

# vectors_inv_scaled = scaler.inverse_transform(vectors_inv_scaled)

# Vectores de soporte (solo del entrenamiento ya que determinan dónde estarán)
plt.scatter(vectores[:, 0], vectores[:, 1], s=200, facecolors="none", edgecolors="red", linewidths=2, label="Vectores de soporte")

niveles = [-1, 0, 1]
niveles_visibles = [nivel for nivel in niveles if Z.min() < nivel < Z.max()]

if len(niveles_visibles) > 0:
    estilos = ["-" if nivel == 0 else "--" for nivel in niveles]
    grosores = [2.5 if nivel == 0 else 1.5 for nivel in niveles]
    plt.contour(xx, yy, Z, levels=niveles_visibles, colors="black", linestyles=estilos, linewidths=grosores)
else:
    print("No hay frontera visible en la región por las asíntotas")

plt.xlabel("Bill lenght")
plt.ylabel("Flipper lenght")

plt.legend()

plt.show()

