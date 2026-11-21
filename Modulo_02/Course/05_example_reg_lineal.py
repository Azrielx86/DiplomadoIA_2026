import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_validate
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

df = pd.DataFrame({
    "Alumno": ["Ana", "Hugo", "Marcos", "Hilda", "Beto", "Lucas", "Georgina", "Octavio"],
    "Study Hours": [1.5, 2.3, 8.0, 4.2, 1.3, 9.6, 3.3, 5.7],
    "Grade": [5.6, 6.1, 7.7, 9.4, 6.3, 9.1, 5.7, 8.3]
})

print(df)

X = df[["Study Hours"]]
y = df["Grade"]

model = LinearRegression()

results = cross_validate(model, X, y, cv=3,
                         scoring={
                             "MAE": "neg_mean_absolute_error",
                             "MSE": "neg_mean_squared_error",
                             "R2": "r2"
                         },
                         return_estimator=True)

print(results)

mae_folds = -results["test_MAE"]
mse_folds = -results["test_MSE"]
rmse_folds = np.sqrt(mse_folds)
rs2_folds = results["test_R2"]

models = results["estimator"]

for i, model_layer in enumerate(models, start=1):
    print(f"[ Fold {i} ]".center(60, '='))
    print(f"├── Model: {model_layer}")
    print(f"├── Equation: {model_layer.coef_[0]:.2f} * Study Hours + {model_layer.intercept_:.2f}")
    print("└── Metrics")
    print(f"\t├── MAE: {mae_folds[i - 1]:.3f}")
    print(f"\t├── MSE: {mse_folds[i - 1]:.3f}")
    print(f"\t├── RMSE: {rmse_folds[i - 1]:.3f}")
    print(f"\t└── R^2: {rs2_folds[i - 1]:.3f}")

print(f"[ Average metrics ]".center(60, '='))
print("└── Metrics")
print(f"\t├── MAE: {mae_folds.mean():.3f}")
print(f"\t├── MSE: {mse_folds.mean():.3f}")
print(f"\t├── RMSE: {rmse_folds.mean():.3f}")
print(f"\t└── R^2: {rs2_folds.mean():.3f}")

final_model = LinearRegression()
final_model.fit(X, y)

y_pred = final_model.predict(X)

print("[ Final Model ]".center(60, '='))
print(y_pred)
print(final_model)
print(f"Equation: {final_model.coef_[0]:.2f} * Study Hours + {final_model.intercept_:.2f}")

df["Prediction"] = y_pred
df["Error"] = y - y_pred
df["Absolute Error"] = df["Error"].abs()
df["Quadratic Error"] = df["Error"] ** 2

print(df)

mae = mean_squared_error(y, y_pred)
mse = mean_squared_error(y, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y, y_pred)

print("Final Metrics")
print(f"\t├── MAE: {mae}")
print(f"\t├── MSE: {mse}")
print(f"\t├── RMSE: {rmse}")
print(f"\t└── R^2: {r2}")

plt.scatter(X["Study Hours"], y, color="blue", label="Grade")
plt.xlabel("Study Hours")
plt.ylabel("Grade")

x_linea = pd.DataFrame({
    "Study Hours": np.linspace(X["Study Hours"].min(), X["Study Hours"].max(), 100),
})

for i, model_fold in enumerate(models, start=1):
    plt.plot(x_linea["Study Hours"], model_fold.predict(x_linea), color=plt.cm.tab10(i % 10), label=f"Fold {i}", linestyle="--")

# Modelo final
plt.plot(x_linea["Study Hours"], final_model.predict(x_linea), color="red", label="Final Model")

plt.legend()
plt.show()