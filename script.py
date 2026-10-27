import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

df = pd.read_csv("tp2_opendata_titanic.csv")
df = df.dropna(subset=["Age", "Embarked"])  # simplification pour le TP
 
features = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]
X = df[features].copy()
X["Sex"] = X["Sex"].map({"male": 0, "female": 1})
X["Embarked"] = X["Embarked"].map({"S": 0, "C": 1, "Q": 2})
y = df["Survived"]
 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=20, random_state=42)
# test_size=20 volontairement petit : suffisant pour un score lisible,
# assez peu pour rester faisable a interroger un LLM ligne par ligne


# Entraîner et scorer le modèle ML

model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)
pred_ml = model.predict(X_test)
 
accuracy_ml = accuracy_score(y_test, pred_ml)
print(f"Score ML (accuracy) : {accuracy_ml:.2f}")