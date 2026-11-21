import matplotlib.pyplot as plt
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score, make_scorer, precision_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB
from sklearn.pipeline import Pipeline

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

# Generación de los sets
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.2, random_state=0, stratify=Y)

model = Pipeline([("vector", CountVectorizer(binary=True)), ("clasificador", BernoulliNB())])

print(f" Resultados ".center(60, "="))

make_scorer(precision_score, pos_label="spam", zero_division=0)
model.fit(X_train, Y_train)

y_pred = model.predict(X_test)

acc = accuracy_score(Y_test, y_pred)
print(f"Accuracy: {acc:.2f}%")

print(" Confusion Matrix ")
cm = confusion_matrix(Y_test, y_pred, labels=["ham", "spam"])
print(cm)

fig, ax = plt.subplots(figsize=(7, 5))

# Coolwarm (Para matrices de correlación)
# 966 - True positive de los mensajes normales que realmente son normales
# 120 - True negative de los mensajes spam que realmente eran spam
# 29 - Correo normal detectado como spam
# 0 - Correo spam detectado como normal
ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["ham", "spam"]).plot(ax=ax, cmap="Purples")

plt.title("Confusion Matrix")
plt.tight_layout()
plt.show()
