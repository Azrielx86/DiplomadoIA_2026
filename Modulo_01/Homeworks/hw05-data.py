"""Tarea 5 - Diplomado de Inteligencia Artificial

Reto práctico:
- Analizar un conjunto de datos temporal.
- Identificar posibles anomalías.
- Visualizar la serie original.
- Aplicar un promedio móvil y comparar resultados.

En el entregable se adjunta la imagen de la gráfica, en azul los datos reales y en naranja los datos promediados.
Como se puede observar en la gráfica azul hay dos picos con mayor afluencia, mientras que casi al final de la gráfica
hay una parte con menor afluencia, estos datos pueden considerarse anormales o fuera del promedio, al aplicar
rolling estos datos entran más en el promedio general.

Equipo:
    - Edgar Chalico
    - Renata Baños

Uso:
    python hw05-data.py -f ../Data/day.csv

    Si se tiene el archivo day.csv en la misma ruta.
    python hw05-data.py
"""

from argparse import ArgumentParser

import matplotlib.pyplot as plt
import pandas as pd

# Esto es para seleccionar el archivo si está en otra ruta diferente a donde se ejecuta el script
parser = ArgumentParser()
parser.add_argument("-f", "--file", required=False)
args = parser.parse_args()

# Filtrado de datos
df = pd.read_csv(args.file or "day.csv")
df["dteday"] = pd.to_datetime(df["dteday"])

start_date = '2011-01-01'
end_date = '2011-01-31'

df = df[df['dteday'].between(start_date, end_date)]

df["cnt_mean"] = df.rolling(window=3, on="dteday")["cnt"].mean()

# Graficado de datos
plt.plot(df["dteday"], df["cnt"])
plt.plot(df["dteday"], df["cnt_mean"])

plt.xticks(rotation=45, ha="right")
plt.xlabel("Day")
plt.ylabel("Rat Count")

plt.show()
