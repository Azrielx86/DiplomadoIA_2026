import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import cross_val_score
import time
from typing import Any
from typing import Dict

from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline
from sklearn.compose import TransformedTargetRegressor
import pandas as pd
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("../casas_taiwan.csv")
print(df.shape)

x = df.drop(columns=["No", "precio"])
y = df["precio"]

model = TransformedTargetRegressor(regressor=make_pipeline(
    StandardScaler(),
    SVR()
), transformer=StandardScaler())

parameters: Dict[str, Any] = {
    "regressor__svr__kernel": ["linear", "poly", "rbf", "sigmoid"],
    "regressor__svr__C": [0.01, 0.1, 1, 10, 100],
    "regressor__svr__epsilon": [0.01, 0.1, 0.5],
    "regressor__svr__gamma": ["scale", "auto"],
}

grid = GridSearchCV(model, parameters, cv=10, scoring="r2", n_jobs=-1)

start = time.time()
grid.fit(x, y)
end = time.time()

print(f"Training time: {end - start:.2f} seconds")
print(f"Best estimator: {grid.best_estimator_}")
print(f"Best estimator parameters: {grid.best_params_}")
print(f"Best r2 (score): {grid.best_score_ * 100:.2f}%")

final_model = grid.best_estimator_

scores = cross_val_score(final_model, x, y, cv=10, scoring="r2", n_jobs=-1)

for i, score in enumerate(scores):
    print(f"Fold {i + 1} | r2 = {score * 100:.2f}%")

y_pred_cv = cross_val_predict(final_model, x, y, cv=10, n_jobs=-1)

mae = mean_absolute_error(y, y_pred_cv)

print(f"MAE = {mae}")

plt.scatter(y, y_pred_cv, alpha=0.7)
limits = [min(min(y), y_pred_cv.min()), max(max(y), y_pred_cv.max())]

plt.plot(limits, limits, "r--")

plt.xlabel("Real price")
plt.ylabel("Predicted price")

plt.show()
