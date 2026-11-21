from sklearn.metrics import r2_score
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pandas as pd

df = pd.read_csv("../casas_taiwan.csv")

X = df[["fecha", "edad", "distancia_metro", "tiendas", "latitud", "longitud"]]
y = df["precio"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = LinearRegression()
model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

# noinspection string-format
print(f"MAE: {mae:.2f}")
# noinspection string-format
print(f"MSE: {mse:.2f}")
# noinspection string-format
print(f"RMSE: {rmse:.2f}")
print(f"R^2: {r2:.2f}")
