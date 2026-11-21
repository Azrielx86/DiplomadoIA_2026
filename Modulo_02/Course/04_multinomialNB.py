import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import ConfusionMatrixDisplay, recall_score
from sklearn.metrics import accuracy_score, make_scorer, precision_score, confusion_matrix
from sklearn.model_selection import cross_validate, train_test_split
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.pipeline import Pipeline

show_graphs = True

columns_tags = ["id", "title", "url", "editorial", "category", "serie1", "serie2", "serie3"]
df = pd.read_csv("../newsCorpora.csv", sep="\t", names=columns_tags)

# Ajuste del dataset
df["title"] = df["title"].str.strip().str.lower()
df["editorial"] = df["editorial"].str.strip().str.lower()
df["url"] = df["url"].str.strip().str.lower()
df["serie1"] = df["serie1"].str.strip().str.lower()
df["serie2"] = df["serie2"].str.strip().str.lower()

df = df.drop(columns=["id"])
X = df['title']
Y = df['category']

print(Y.value_counts())

model = Pipeline([("vector", CountVectorizer(binary=True)), ("clasificador", MultinomialNB())])

scoring_params = {
    "accuracy": "accuracy",
    "precision": "precision_macro",
    "recall": "recall_macro"
}

results = cross_validate(model, X, Y, cv=100, scoring=scoring_params, return_train_score=True, n_jobs=16)

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
    ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["e", "b", "t", "m"]).plot(ax=ax, cmap="Purples")
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
