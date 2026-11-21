"""Edgar Ulises Moreno Chalico - 318192950
Tarea 01 - Diplomado ML Facultad de Ingeniería

1. Aplica análisis y exploración de datos al dataset WineDataset.csv, realiza 8 gráficas
con seaborn y añade una explicación de lo que representa
Más información del dataset en: https://archive.ics.uci.edu/dataset/109/wine

2. Aplica KNN a este dataset. Usa cv de 5 folds y GridSearchCV.¿De cuánto es el
accuracy? Aunque los datos tengan variabilidad, todavía no escales.

Fecha de entrega: 10 de octubre antes de medianoche
La tarea se subirá a classroom.

"""

import pandas as pd

from argparse import ArgumentParser

from matplotlib import pyplot as plt
import seaborn as sns

parser = ArgumentParser()
parser.add_argument("-f", "--file", required=False, default="./WineDataset.csv", help="path to file")

args = parser.parse_args()

file_path: str = args.file

attributes = [
    "Class",
    "Alcohol",
    "Malic acid",
    "Ash",
    "Alcalinity of ash",
    "Magnesium",
    "Total phenols",
    "Flavanoids",
    "Nonflavanoid phenols",
    "Proanthocyanins",
    "Color intensity",
    "Hue",
    "OD280/OD315 of diluted wines",
    "Proline"
]

df = pd.read_csv(file_path, sep=",", names=attributes)

print(" Data head ".center(80, "="))
print(df.head())

# Gráficas
for att in attributes[1:]:
    sns.boxplot(data=df, x="Class", y=att, hue="Class")
    plt.title(f"Variabilidad de {att}")
    plt.xlabel("Clase")
    plt.ylabel("Alcohol")
    plt.tight_layout()
    plt.show()