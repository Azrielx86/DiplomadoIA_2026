import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from keras import Input
from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

df = sns.load_dataset("penguins")

df.info()

columns = ["bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g"]

df = df.dropna(subset=columns)

x = df[columns]
y = df["species"]

encoder = LabelEncoder()
y = encoder.fit_transform(y)

# Entrenamiento y prueba
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42, stratify=y)

# Separación para la validación
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=42, stratify=y_train)

scaler = StandardScaler()

x_train = scaler.fit_transform(x_train)
x_val = scaler.transform(x_val)
x_test = scaler.transform(x_test)

model = Sequential(
    [
        Input(shape=(4,)),
        Dense(16, activation='relu'),
        Dense(8, activation='relu'),
        Dense(3, activation='softmax')
    ]
)

model.compile(optimizer=Adam(learning_rate=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

historicos = model.fit(x_train, y_train, batch_size=10, epochs=100, verbose=1, validation_data=(x_val, y_val))

loss_train, accuracy_train = model.evaluate(x_train, y_train, verbose=1)
loss_test, accuracy_test = model.evaluate(x_test, y_test, verbose=1)
loss_val, accuracy_val = model.evaluate(x_val, y_val, verbose=1)

print("[ Resultados ]".center(50, '='))
print(f"Accuracy Train: {accuracy_train * 100:.4f}%")
print(f"Accuracy Val: {accuracy_val * 100:.4f}%")
print(f"Accuracy Test: {accuracy_test * 100:.4f}%")

print(f"Loss Train: {loss_train * 100:.4f}%")
print(f"Loss Validation: {loss_val * 100:.4f}%")
print(f"Loss Test: {loss_test * 100:.4f}%")

# Gráfica de la función de pérdida
plt.plot(historicos.history['loss'], label='Train')
plt.plot(historicos.history['val_loss'], label='Validation')
plt.title("Loss: Train vs. Validation")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()

# Gráfica de Accuracy
plt.plot(historicos.history['accuracy'], label='Train')
plt.plot(historicos.history['val_accuracy'], label='Validation')
plt.title("Accuracy: Train vs. Validation")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()

diff = accuracy_train - accuracy_val
print(f"Difference Train: {diff * 100:.4f}%")
print("Sobre ajuste" if diff > 0.09 else "Sin sobre ajuste")

loss_val_hist = np.array(historicos.history["val_loss"])
mejor_epoca = np.argmin(loss_val_hist) + 1
print("Mejor época: ", mejor_epoca)
