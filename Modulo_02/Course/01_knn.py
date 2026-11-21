import time

import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier

df = pd.read_csv('../irisData.csv')

print(df.shape)

x = df.iloc[:, 0:4]
y = df.iloc[:, -1]

knn = KNeighborsClassifier()

parameters = {
    "n_neighbors": [5, 7, 10, 15, 20],
    "weights": ["uniform", "distance"],
    "metric": ["minkowski", "manhattan"]
}

grid = GridSearchCV(estimator=knn,
                    param_grid=parameters,
                    cv=10,
                    scoring='accuracy')

start = time.time()
grid.fit(x, y)
end = time.time()

print(f"Best hyperparameters: {grid.best_params_}")
print(f"Best Accuracy: {grid.best_score_:.2f}%")
print(f"Best score: {grid.best_score_:.2f}")
print(f"Training time: {end - start:.2f} seconds")

new_flower = pd.DataFrame([[3.7, 1.8, 3.9, 1.1]], columns=x.columns)
result = grid.predict(new_flower)
print(f"New flower: {result}")
