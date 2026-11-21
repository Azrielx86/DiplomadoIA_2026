from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import CategoricalNB
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split

import pandas as pd

show_graphs = True

df = pd.read_csv("../mushrooms.csv")
print(df.head())
print(f"Shape of the DataFrame: {df.shape}")

X = df.drop(columns=["class"]).copy()
y = df['class']

# Separar en entrenamiento y prueba sin validación cruzada
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)

X_train_cod = encoder.fit_transform(X_train).astype(int) + 1
X_test_cod = encoder.transform(X_test).astype(int) + 1

model = CategoricalNB(min_categories=2)

model.fit(X_train_cod, y_train)

y_pred = model.predict(X_test_cod)

print(f"Accuracy score: {accuracy_score(y_test, y_pred) * 100:.3f}%")

cm = confusion_matrix(y_test, y_pred, labels=['e', 'p'])

if show_graphs:
    fig, ax = plt.subplots(figsize=(7, 5))
    ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['e', 'p']).plot(ax=ax, cmap="Purples")
    plt.title("Confusion Matrix")
    plt.tight_layout()

    # Graficar comparativa de entrenamiento, test para cada fold
    # plt.figure(figsize=(7, 5))
    # plt.plot(range(1, 11), results["train_accuracy"], label="Training")
    # plt.plot(range(1, 11), results["test_accuracy"], label="Test")
    # plt.title("Accuracy per Fold")
    # plt.legend()
    # plt.tight_layout()

    plt.show()