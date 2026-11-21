import matplotlib.pyplot as plt
import numpy as np
from keras.datasets import mnist
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense
from keras.models import Sequential
from keras.optimizers import Adam
from sklearn.metrics import accuracy_score, ConfusionMatrixDisplay

image_size = 28
(x_train, y_train), (x_test, y_test) = mnist.load_data()
print(f"Training: {x_train.shape}")
print(f"Testing: {x_test.shape}")

x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)

model = Sequential(
    [
        Input(shape=(28, 28, 1)),
        # Inicio de una convolución
        Conv2D(filters=64, kernel_size=(3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        # Fin de una convolución
        Conv2D(filters=128, kernel_size=(3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Flatten(),
        Dense(units=128, activation='relu'),
        Dense(units=10, activation='softmax')
    ]
)

model.compile(optimizer=Adam(learning_rate=0.01), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

historical = model.fit(x_train, y_train, batch_size=64, validation_split=0.2, epochs=10, verbose=1)

loss, accuracy = model.evaluate(x_test, y_test)

print(f"Loss: {loss:.4f} | Accuracy: {accuracy:.4f}")

# Probabilidades de las clases
probabilities = model.predict(x_test)
y_pred = np.argmax(probabilities, axis=1)
print(F"Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")

ConfusionMatrixDisplay.from_predictions(y_test, y_pred, display_labels=range(10), cmap="Reds")
plt.title("Confusion Matrix MNIS with CNN")
plt.tight_layout()
plt.show()
