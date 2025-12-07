import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import joblib

# RANDOM SEED
seed = 42

# READ DATASET
df = pd.read_csv("Lifestyle_and_Health_Risk_Prediction_Synthetic_Dataset.csv")

# shuffle data
df = df.sample(frac=1, random_state=seed)

# FEATURES & TARGET
features = ['age', 'bmi', 'smoking', 'alcohol', 'sleep', 'sugar_intake']
X = df[features].apply(pd.to_numeric, errors='coerce')
y = df['health_risk']

# DROP NaN (IMPORTANT)
mask = X.notna().all(axis=1)
X = X[mask]
y = y[mask]

# TRAIN TEST SPLIT
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=seed,
    stratify=y
)

# RANDOM FOREST
rf_model = RandomForestClassifier(
    n_estimators=100,
    random_state=seed
)
rf_model.fit(X_train, y_train)

rf_pred = rf_model.predict(X_test)
rf_acc = accuracy_score(y_test, rf_pred)
print(f"Random Forest Accuracy: {rf_acc:.3f}")

# KNN (WITH SCALING)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train_scaled, y_train)

knn_pred = knn_model.predict(X_test_scaled)
knn_acc = accuracy_score(y_test, knn_pred)
print(f"KNN Accuracy: {knn_acc:.3f}")

# SAVE MODELS
joblib.dump(rf_model, "rf_model.sav")
joblib.dump(knn_model, "knn_model.sav")
joblib.dump(scaler, "scaler.sav")
