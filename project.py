
# ==========================================================
# COMMODITY PRICE ESTIMATION USING MACHINE LEARNING
# Algorithm: Random Forest Regression
# ==========================================================

# 1. IMPORT LIBRARIES

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score


# 2. LOAD DATASET

df = pd.read_csv(
    r"C:/Users/hp/Downloads/mandi_prices_with_simulated_dates.csv"
)

print("\nDATASET LOADED SUCCESSFULLY")
print("Dataset Shape:", df.shape)
print("\nFirst Five Records:")
print(df.head())

print("\nDataset Information:")
print(df.info())


# 3. DATA PREPROCESSING

# Convert modal price into numeric format
target = "Modal_x0020_Price"

df[target] = pd.to_numeric(
    df[target], errors="coerce"
)

# Remove missing and invalid target prices
df = df.dropna(subset=[target])
df = df[df[target] > 0].copy()

# Remove duplicate records
df = df.drop_duplicates()

# Select input features
features = [
    "State",
    "District",
    "Market",
    "Commodity",
    "Variety",
    "Grade"
]

# Handle missing categorical values
for column in features:
    df[column] = (
        df[column]
        .fillna("Unknown")
        .astype(str)
        .str.strip()
    )

# Target variable
X = df[features]
y = df[target]

print("\nDATA PREPROCESSING COMPLETED")
print("Records:", len(df))
print("Input Features:", features)
print("Target Variable:", target)

print("\nMissing Values:")
print(df[features + [target]].isnull().sum())


# 4. EXPLORATORY DATA ANALYSIS

print("\nPRICE STATISTICS")
print(df[target].describe())

# Top 10 commodities by average modal price
top_commodities = (
    df.groupby("Commodity")[target]
    .mean()
    .nlargest(10)
    .sort_values()
)

plt.figure(figsize=(10, 6))
top_commodities.plot(kind="barh")
plt.title("Top 10 Commodities by Average Modal Price")
plt.xlabel("Average Modal Price")
plt.ylabel("Commodity")
plt.tight_layout()
plt.show()

# Modal price distribution
plt.figure(figsize=(8, 5))

plt.hist(df["Modal_x0020_Price"], bins=20)

plt.title("Distribution of Modal Market Prices")
plt.xlabel("Modal Price")
plt.ylabel("Number of Records")

plt.show()


# 5. TRAINING AND TESTING SPLIT

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("\nTRAINING RECORDS:", len(X_train))
print("TESTING RECORDS:", len(X_test))


# 6. ENCODING CATEGORICAL DATA

preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            features
        )
    ]
)


# 7. CREATE RANDOM FOREST MODEL

model = Pipeline([
    ("preprocessor", preprocessor),

    ("regressor", RandomForestRegressor(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    ))
])


# 8. TRAIN THE MODEL

print("\nMODEL TRAINING STARTED...")

model.fit(X_train, y_train)

print("MODEL TRAINING COMPLETED!")


# 9. PREDICT MODAL MARKET PRICES

predictions = model.predict(X_test)

print("\nACTUAL VS PREDICTED PRICES")

comparison = pd.DataFrame({
    "Actual_Price": y_test.values,
    "Predicted_Price": np.round(predictions, 2)
})

print(comparison.head(10))


# 10. MODEL EVALUATION

mae = mean_absolute_error(y_test, predictions)

rmse = np.sqrt(
    mean_squared_error(y_test, predictions)
)

r2 = r2_score(y_test, predictions)

print("\nMODEL PERFORMANCE")
print("--------------------------------")
print("MAE:", round(mae, 2))
print("RMSE:", round(rmse, 2))
print("R2 Score:", round(r2, 3))


# 11. ACTUAL VS PREDICTED GRAPH

plt.figure(figsize=(8, 6))

plt.scatter(
    y_test,
    predictions,
    alpha=0.5
)

plt.xlabel("Actual Modal Price")
plt.ylabel("Predicted Modal Price")
plt.title("Actual vs Predicted Commodity Prices")

plt.tight_layout()
plt.show()


# 12. SAMPLE PRICE PREDICTION

# Use a real test record as an example
sample = X_test.iloc[[0]]

predicted_price = model.predict(sample)[0]

print("\nSAMPLE PRICE PREDICTION")
print("--------------------------------")

print("Commodity Details:")
print(sample.to_string(index=False))

print(
    "\nEstimated Modal Market Price:",
    round(predicted_price, 2)
)


# 13. SAVE PREDICTION RESULTS

results = pd.DataFrame({
    "Actual_Price": y_test.values,
    "Predicted_Price": predictions
})

results.to_csv(
    "commodity_price_predictions.csv",
    index=False
)

print("\nPREDICTIONS SAVED SUCCESSFULLY!")
print("File: commodity_price_predictions.csv")

print("\nPROJECT COMPLETED SUCCESSFULLY!")
