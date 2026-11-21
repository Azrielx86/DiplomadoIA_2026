from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import StandardScaler

df = pd.read_csv('../titanicData.csv')

print("[Original Data]".center(60, '='))
print(f"Data shape: {df.shape}")
df.info()

# Borrar variables que no se usarán
df = df.drop(columns=['PassengerId', 'Name', 'Ticket', 'Cabin'])

# Llenado de valores incompletos
df['Age'] = df['Age'].fillna(df['Age'].mean())
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])

# Edades promedio por clase
# age_per_class_avg = df.groupby('Pclass')['Age'].mean()

print("[Fixed Data]".center(60, '='))
print(f"Data shape: {df.shape}")
df.info()

# sns.countplot(data=df, x='Sex', hue='Survived')
# plt.show()

X = df.drop(columns=['Survived']).copy()
y = df['Survived']

print("[ Data original ]".center(60, '='))
print(X)
print(y)

# Convertir categóricos a numéricos

X = pd.get_dummies(X, columns=["Sex", "Embarked"], dtype=int)

print("[ Data fixed ]".center(60, '='))
print(X)

print("[ Data scaling ]".center(60, '='))
scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

model = LogisticRegression(solver='liblinear')

model.fit(X_scaled, y)

results = cross_val_score(model, X_scaled, y, cv=5, scoring='accuracy')

print("[Accuracy on each fold]".center(60, '='))
for i, accuracy in enumerate(results, start=1):
    print(f"Layer {i}: {accuracy * 100:.2f}%")

print(f"Average accuracy: {results.mean() * 100:.2f}%")

# Which layer will be used for prediction?
y_pred_cv = cross_val_predict(model, X_scaled, y, cv=5)

matrix = confusion_matrix(y, y_pred_cv)

print("[Confusion Matrix]".center(60, '='))
print(matrix)

sns.heatmap(matrix, annot=True, cmap='Blues', fmt='d',
            xticklabels=['No Sobrevivió, Sobrevivió'],
            yticklabels=['No Sobrevivió, Sobrevivió'])
plt.show()

print(classification_report(y, y_pred_cv))

model.fit(X_scaled, y)

new_passenger = [{
    'Pclass': 3,
    'Age': 28,
    'SibSp': 1,
    'Parch': 2,
    'Fare': 50.83,
    'Sex_female': 1,
    'Sex_male': 0,
    'Embarked_C': 0,
    'Embarked_Q': 1,
    'Embarked_S': 0
}]

new_passenger = pd.DataFrame(new_passenger, columns=X.columns)
new_passenger = scaler.transform(new_passenger)

prediction = model.predict(new_passenger)
print(f'Answer: {prediction[0]} ({"Survived" if prediction[0] == 1 else "Not Survived"})')

probability = model.predict_proba(new_passenger)

print(f'Probability: {probability[0]}')

print(f'Survival Probability: {probability[0][1] * 100:.2f}%')
print(f'Dead Probability: {probability[0][0] * 100:.2f}%')
