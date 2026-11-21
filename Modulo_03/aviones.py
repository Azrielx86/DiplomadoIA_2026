import pandas as pd
from keras import Sequential, Input
from keras.layers import Dense
from keras.optimizers import Adam
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("../Datasets/AirfoilSelfNoise.csv")

print(df.info())

x = df.drop(columns=['SSPL'])
y = df['SSPL']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=42)

scaler = StandardScaler()

x_train = scaler.fit_transform(x_train)
x_val = scaler.transform(x_val)
x_test = scaler.transform(x_test)

model = Sequential(
    [
        Input(shape=(5,)),
        Dense(30, activation='softplus'),
        Dense(15, activation="relu"),
        Dense(1, activation='linear'),
    ]
)

model.compile(optimizer=Adam(learning_rate=0.001), loss='mae', metrics=['mae'])

model.fit(x_train, y_train, epochs=200, verbose=1, validation_data=(x_val, y_val))

y_pred = model.predict(x_test)

r2 = r2_score(y_test, y_pred)
print(f"R2: {r2}")