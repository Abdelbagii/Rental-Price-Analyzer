import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score


os.makedirs("models", exist_ok=True)


# =========================
# LOAD DATASET
# =========================

df = pd.read_csv("data/housing_train.csv")


# =========================
# CLEAN DATA
# =========================

useful_columns = [
    "price",
    "region",
    "state",
    "type",
    "sqfeet",
    "beds",
    "baths",
    "cats_allowed",
    "dogs_allowed",
    "smoking_allowed",
    "wheelchair_access",
    "electric_vehicle_charge",
    "comes_furnished",
    "laundry_options",
    "parking_options",
    "lat",
    "long"
]

df = df[useful_columns].copy()

df = df.dropna()

# Remove unrealistic values
df = df[df["price"] > 100]
df = df[df["price"] < 10000]
df = df[df["sqfeet"] > 100]
df = df[df["sqfeet"] < 10000]
df = df[df["beds"] >= 0]
df = df[df["beds"] <= 10]
df = df[df["baths"] > 0]
df = df[df["baths"] <= 10]

target = "price"

X = df.drop(columns=[target])
y = df[target]


# =========================
# ENCODE CATEGORICAL DATA
# =========================

X = pd.get_dummies(X, drop_first=True)

columns = X.columns.tolist()


# =========================
# TRAIN TEST SPLIT
# =========================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# =========================
# TRAIN MODEL
# =========================

model = RandomForestRegressor(
    n_estimators=150,
    max_depth=18,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)


# =========================
# EVALUATE MODEL
# =========================

y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Model Training Completed")
print("-" * 30)
print("Mean Absolute Error:", round(mae, 2))
print("R2 Score:", round(r2, 4))


# =========================
# SAVE MODEL FILES
# =========================

joblib.dump(model, "models/rent_model.pkl")
joblib.dump(columns, "models/columns.pkl")

print("Model and columns saved successfully.")