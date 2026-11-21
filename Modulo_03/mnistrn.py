import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # Ocultar logs INFO y WARNING de TensorFlow
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1" # No usar CUDA

import time

import numpy as np
import matplotlib.pyplot as plt
import keras
import pandas as pd
import seaborn as sns

from keras import Input
from keras.layers import Dense, Flatten
from keras.models import Sequential
from keras.optimizers import Adam
from keras.utils import to_categorical
from scikeras.wrappers import KerasClassifier
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score, precision_score, recall_score, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

show_info_graphs = False

# Cargar el dataset
(X_train_full, y_train_full), (X_test_full, y_test_full) = keras.datasets.mnist.load_data()
print(f"Entrenamiento: {X_train_full.shape}")
print("Distribución de clases")
print(pd.Series(y_train_full).value_counts().sort_index())

if show_info_graphs:
    sns.countplot(x=y_train_full)
    plt.title("Distribución de los dígitos")
    plt.xlabel("Dígito")
    plt.ylabel("Cantidad de apariciones")
    plt.show()

    for i in range(5):
        plt.imshow(X_train_full[i], cmap='gray')
        plt.title(f"Dígito: {y_train_full[i]}")
        plt.axis('off')
        plt.show()

X_train_full = X_train_full.astype("float32") / 255.0
X_test = X_test_full.astype("float32") / 255.0

X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=0.2, random_state=42,
                                                  stratify=y_train_full)

model = Sequential(
    [
        Input(shape=(28, 28)),
        Flatten(),
        Dense(300, activation='softplus'),
        Dense(300, activation='relu'),
        Dense(10, activation='softmax'),
    ]
)

model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy'],
)

historicos = model.fit(X_train, y_train, batch_size=1000, epochs=100, verbose=1, validation_data=(X_val, y_val))

plt.plot(historicos.history['accuracy'], label="Train Accuracy")
plt.plot(historicos.history['val_accuracy'], label="Validation Accuracy")
plt.title("Accuracy: Train vs. Validation")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.show()

plt.plot(historicos.history['loss'], label="Train Loss")
plt.plot(historicos.history['val_loss'], label="Validation Loss")
plt.title("Loss: Train vs. Validation")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()

probabilities = model.predict(X_test)
y_pred = np.argmax(probabilities, axis=1)

print("Métricas de desempeño")
print(f"Accuracy: {accuracy_score(y_test_full, y_pred)}")
print(f"Precision: {precision_score(y_test_full, y_pred, average='macro', zero_division=0)}")
print(f"Recall: {recall_score(y_test_full, y_pred, average='macro', zero_division=0)}")
