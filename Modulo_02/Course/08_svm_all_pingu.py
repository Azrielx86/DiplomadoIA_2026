import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, classification_report, precision_score, \
    recall_score

df = sns.load_dataset("penguins")

print(df.shape)
df = df.dropna()
df.info()

x = df.drop(columns=["species"]).copy()
y = df["species"].copy()

categorical_columns = x.select_dtypes(include=["object", "str"]).columns.tolist()
numerical_columns = x.select_dtypes(include=["number"]).columns.tolist()

procesador_numerico = Pipeline([
    ("imputador", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])

procesador_categorico = Pipeline([
    ("imputador", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore")),
])

processor = ColumnTransformer([
    ("numericas", procesador_numerico, numerical_columns),
    ("categoricas", procesador_categorico, categorical_columns),
])

model = Pipeline([
    ("preprocessing", processor),
    ("svm", SVC())
])

parameters = [
    {
        "svm__kernel": ['linear'],
        "svm__C": [0.01, 0.1, 1, 10, 100],
    },
    {
        "svm__kernel": ['poly'],
        "svm__C": [0.01, 0.1, 1, 10, 100],
        "svm__degree": [2, 3, 4, 5],
        "svm__gamma": ["scale", "auto"],
        "svm__coef0": [0, 1, 2]
    },
    {
        "svm__kernel": ['rbf'],
        "svm__C": [0.01, 0.1, 1, 10, 100],
        "svm__gamma": [0.01, 0.1, 1, 10],
    },
    {
        "svm__kernel": ['sigmoid'],
        "svm__C": [0.01, 0.1, 1, 10, 100],
        "svm__gamma": [0.01, 0.1, 1, 10],
        "svm__coef0": [-1, 0, 1],
    }
]

grid = GridSearchCV(estimator=model,
                    param_grid=parameters,
                    cv=10,
                    scoring="accuracy",
                    return_train_score=True,
                    n_jobs=-1)

grid.fit(x, y)

print("[ Results ]".center(60, '='))
print(f"Best estimator: {grid.best_estimator_}")
print(f"Best params: {grid.best_params_}")
print(f"Best score: {grid.best_score_}")

print("[ Modelo final ]".center(60, '='))

final_model = grid.best_estimator_

# Hacer predict al modelo final

