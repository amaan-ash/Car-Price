import pandas as pd
import joblib

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    root_mean_squared_error
)

# =====================================================
# Load Dataset
# =====================================================

df = pd.read_csv("dataset/CAR DETAILS FROM CAR DEKHO.csv")

# =====================================================
# Feature Engineering
# =====================================================

df["brand"] = df["name"].str.split().str[0]

CURRENT_YEAR = 2026

df["car_age"] = CURRENT_YEAR - df["year"]

# -----------------------------------------------------
# Save metadata BEFORE encoding
# -----------------------------------------------------

metadata = {

    "brands": sorted(df["brand"].unique().tolist()),

    "fuel":
        sorted(df["fuel"].unique().tolist()),

    "seller_type":
        sorted(df["seller_type"].unique().tolist()),

    "transmission":
        sorted(df["transmission"].unique().tolist()),

    "owner":
        sorted(df["owner"].unique().tolist()),

    "current_year":
        CURRENT_YEAR
}

joblib.dump(metadata,"model/metadata.pkl")

# =====================================================
# Remove unwanted columns
# =====================================================

df.drop(["name","year"],axis=1,inplace=True)

# =====================================================
# Label Encoding
# =====================================================

encoders={}

categorical_columns=[

    "brand",
    "fuel",
    "seller_type",
    "transmission",
    "owner"
]

for column in categorical_columns:

    encoder=LabelEncoder()

    df[column]=encoder.fit_transform(df[column])

    encoders[column]=encoder

joblib.dump(encoders,"model/encoders.pkl")

# =====================================================
# Features
# =====================================================

X=df.drop("selling_price",axis=1)

y=df["selling_price"]

# =====================================================
# Split
# =====================================================

X_train,X_test,y_train,y_test=train_test_split(

    X,
    y,

    test_size=0.20,

    random_state=42
)

# =====================================================
# Model
# =====================================================

model=RandomForestRegressor(

    n_estimators=500,

    max_depth=20,

    random_state=42,

    n_jobs=-1
)

model.fit(X_train,y_train)

# =====================================================
# Prediction
# =====================================================

prediction=model.predict(X_test)

# =====================================================
# Metrics
# =====================================================

r2=r2_score(y_test,prediction)

mae=mean_absolute_error(y_test,prediction)

rmse=root_mean_squared_error(y_test,prediction)

print("\n")

print("="*60)

print("MODEL PERFORMANCE")

print("="*60)

print(f"R² Score : {r2:.4f}")

print(f"MAE      : {mae:.2f}")

print(f"RMSE     : {rmse:.2f}")

joblib.dump(model,"model/car_price_model.pkl")

print("\nModel Saved Successfully")