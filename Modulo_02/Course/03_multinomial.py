import matplotlib.pyplot as plt
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score, make_scorer, precision_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB
from sklearn.pipeline import Pipeline

columns_tags = ["id", "title", "url", "editorial", "category", "serie1", "serie2", "serie3"]
df = pd.read_csv("../newsCorpora.csv", sep="\t", names=columns_tags)

# Ajuste del dataset
df["title"] = df["title"].str.strip().str.lower()
df["editorial"] = df["editorial"].str.strip().str.lower()
df["url"] = df["url"].str.strip().str.lower()
df["serie1"] = df["serie1"].str.strip().str.lower()
df["serie2"] = df["serie2"].str.strip().str.lower()

print(f"Shape: {df.shape}")

print(" Data with nulls ".center(60, "="))
print(df.isna().sum())

print(" Different Categories ".center(60, "="))
print(df["category"].value_counts())

X = df.drop(columns=["category", "id"]).copy()
Y = df['category']

X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.2, random_state=0, stratify=Y)