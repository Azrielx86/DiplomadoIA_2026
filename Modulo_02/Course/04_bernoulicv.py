import numpy as np
from sklearn.model_selection import cross_validate
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import ConfusionMatrixDisplay, recall_score
from sklearn.metrics import accuracy_score, make_scorer, precision_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB
from sklearn.pipeline import Pipeline

show_graphs = False

df = pd.read_csv("../spam.csv")
print(f"Data shape: {df.shape}")

print(" Data Head ".center(60, "-"))
print(df.head())

print(" Data info ".center(60, "-"))
df.info()

df["mensaje"] = df["mensaje"].str.strip().str.lower()
df["target"] = df["target"].str.strip().str.lower()

print(" Value counts ".center(60, "-"))
print(df["target"].value_counts())

X = df["mensaje"]
Y = df["target"]

model = Pipeline([("vector", CountVectorizer(binary=True)), ("clasificador", BernoulliNB())])

print(f" Resultados ".center(60, "="))

scoring_params = {
    "accuracy": "accuracy",
    "precision": make_scorer(precision_score, pos_label="spam", zero_division=0),
    "recall": make_scorer(recall_score, pos_label="spam", zero_division=0)
}

results = cross_validate(model, X, Y, cv=10, scoring=scoring_params, return_train_score=True, n_jobs=16)

print(" Fold and their metrics ".center(60, "-"))

for i in range(10):
    print(f"==> Fold {i + 1}")
    print(f"Accuracy: {results['test_accuracy'][i]}")
    print(f"Precision: {results['test_precision'][i]}")
    print(f"Recall: {results['test_recall'][i]}")

print(" Mean Performance ".center(60, "="))
print(f"Accuracy: {np.mean(results['test_accuracy'])}")
print(f"Precision: {np.mean(results['test_precision'])}")
print(f"Recall: {np.mean(results['test_recall'])}")

# Análisis de sobreajuste

accuracy_train = np.mean(results['train_accuracy'])
accuracy_cv = np.mean(results['test_accuracy'])

print(" Data to calculate over-adjust ".center(60, "="))
print(f"Training: {accuracy_train}")
print(f"Validation: {accuracy_cv}")
print(f"Diff: {accuracy_train - accuracy_cv}")


print(" Confusion Matrix ".center(60, "="))

model.fit(X, Y)

y_pred = model.predict(X)

cm = confusion_matrix(Y, y_pred, labels=["ham", "spam"])
print(cm)

if show_graphs:
    fig, ax = plt.subplots(figsize=(7, 5))
    ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["ham", "spam"]).plot(ax=ax, cmap="Purples")
    plt.title("Confusion Matrix")
    plt.tight_layout()

    # Graficar comparativa de entrenamiento, test para cada fold
    plt.figure(figsize=(7, 5))
    plt.plot(range(1, 11), results["train_accuracy"], label="Training")
    plt.plot(range(1, 11), results["test_accuracy"], label="Test")
    plt.title("Accuracy per Fold")
    plt.legend()
    plt.tight_layout()

    plt.show()

# Predecir correo nuevo
print(" Test messages ".center(60, "="))
new_messages = [
    "You won the Nobel Prize!",
    "You won $10000 dollars, reply to this email NOW!",
    "Hi honey, I've sent you a pic! ;)",
    "Hi, are you available tomorrow at 12?",
    "I have a terminal illness, I have a lot of money and I don't know whom can have it, you may be the ideal person, please, contact me so I can send my money, about $1.5M dollars, I know you will use it correctly. Regards."
]

predictions = model.predict(new_messages)

for message, prediction in zip(new_messages, predictions):
    print(f"=> Message: {message}")
    print(f"Prediction: {prediction}")
