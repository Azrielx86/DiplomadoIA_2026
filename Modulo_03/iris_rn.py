import time

import numpy as np
import matplotlib.pyplot as plt

from keras import Input
from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import Adam
from keras.utils import to_categorical
from scikeras.wrappers import KerasClassifier
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score, precision_score, recall_score, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def base_model() -> Sequential:
    model = Sequential([
        Input(shape=(4,)),  # Valores de las características, se esperan 4 características
        Dense(8, activation='relu'),  # Primer capa oculta, esta define la cantidad de neuronas en la primera capa
        Dense(8, activation='softplus'),  # Segunda capa oculta
        Dense(3, activation='softmax'),  # Capa de salida
    ])

    model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

    return model


# Cargar dataset
iris = load_iris()

# pyrefly: ignore[missing-attribute]
X = iris.data
# pyrefly: ignore[missing-attribute]
y = iris.target

# División de datos
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Normalización
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

y_train_cat = to_categorical(y_train, num_classes=3)

classifier = KerasClassifier(model=base_model(), batch_size=8, verbose=1, random_state=42, epochs=200)

start = time.time()
classifier.fit(X_train, y_train_cat)
end = time.time()

probability_test = classifier.model_.predict(X_test)

y_pred = np.argmax(probability_test, axis=1)

print("[ Results ]".center(60, '='))

print(f"Accuracy: {accuracy_score(y_test, y_pred) * 100}%")
print(f"Precision: {precision_score(y_test, y_pred, average='macro', zero_division=0) * 100}%")
print(f"Recall: {recall_score(y_test, y_pred, average='macro', zero_division=0) * 100}%")
print(f"Time train: {end - start}s")

# pyrefly: ignore[missing-attribute]
ConfusionMatrixDisplay.from_predictions(y_test, y_pred, labels=[0, 1, 2], display_labels=iris.target_names)
plt.title('Confusion Matrix')
plt.show()
