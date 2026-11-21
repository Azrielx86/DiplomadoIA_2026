import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report


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

C_global = 0.001

X = np.vstack((adelie, gentoo))
y = np.array([-1] * 15 + [1] * 15)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)

model = SVC(kernel='linear', C=C_global)

model.fit(X_train, y_train)

W = model.coef_[0]
b = model.intercept_[0]

print(f"Ecuación de la recta o frontera: {W[0]} * bill_lenght + {W[1]} * flipper_lenght + {b}")

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
    elif product > 0 and product < 1:
        state = "Dentro del márgen bien clasificado"
    else:
        state = "No viola el márgen"

    print(f"===> Pinguino: {i + 1} | Clase real: {y_train[i]} | f(x): {f[i]} | {state =}")


# J = 1/2 ||W|| + C * sigma(shi)

norma_W_cuadrado = np.dot(W, W)
termino_margen = 0.5 * norma_W_cuadrado
termino_errores = C_global * shi.sum()

J = termino_margen + termino_errores

print(f"J = 1/2 * {termino_margen} + {C_global} * sigma({termino_errores})")

print(f"Vectores de soporte ({len(model.support_vectors_)}):")
print(model.support_vectors_)

print(classification_report(y_test, y_pred_test, labels=[-1, 1]))

# Graficando el entorno del modelo

x_min = X[:, 0].min() - 3
x_max = X[:, 0].max() + 3

y_min = X[:, 1].min() - 3
y_max = X[:, 1].max() + 3

xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300), np.linspace(y_min, y_max, 300))


puntos = np.c_[xx.ravel(), yy.ravel()]

print(puntos)

Z = model.decision_function(puntos)
Z = Z.reshape(xx.shape)

plt.contourf(xx, yy, Z, levels=[-1000, 0, 1000], colors=["royalblue", "seagreen"], alpha=0.12)

# Pintar a los adelie (training)

plt.scatter(X_train[y_train == -1, 0], X_train[y_train == -1, 1], color="royalblue", s=70, label="Adelie (training)")
plt.scatter(X_train[y_train == 1, 0], X_train[y_train == 1, 1], color="seagreen", s=70, label="Gentoo (training)")

# Pintar a los de prueba
plt.scatter(X_test[y_test == -1, 0], X_test[y_test == -1, 1], color="royalblue", s=120, edgecolors="black", linewidths=2, label="Adelie (test)")
plt.scatter(X_test[y_test == 1, 0], X_test[y_test == 1, 1], color="seagreen", s=120, edgecolors="black", linewidths=2, label="Gentoo (test)")

plt.contour(xx, yy, Z, levels=[-1, 0, 1], colors=["black"], linestyles=['--', '-', '--'], linewidths=[1.5, 2.5, 1.5])

# Vectores de soporte (solo del entrenamiento ya que determinan dónde estarán)
plt.scatter(model.support_vectors_[:, 0], model.support_vectors_[:, 1], s=200, facecolors="none", edgecolors="red", linewidths=2, label="Vectores de soporte")

plt.xlabel("Bill lenght")
plt.ylabel("Flipper lenght")

plt.legend()

plt.show()

