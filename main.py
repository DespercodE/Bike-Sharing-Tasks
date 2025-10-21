
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from ucimlrepo import fetch_ucirepo

# fetch dataset
bike_sharing = fetch_ucirepo(id=275)

# data (as pandas dataframes)
X = bike_sharing.data.features
y = bike_sharing.data.targets
# combination of X and y for full view
df = pd.concat([X, y], axis=1)


print(X.head())
print(y.head())
print(bike_sharing.metadata)
print(bike_sharing.variables)

# d1 tasks
print("\nFirst 5 rows:")
print(df.head())

print("\nSummary statistics:")
print(df.describe())

print("\nDataset info:")
print(df.info())

# checking for missing values
print("\nNumber of missing values:")
print(df.isnull().sum())

# defining columns to encode
categorical_columns = ["season", "mnth", "weekday", "weathersit"]
df_encoded = pd.get_dummies(df, columns=categorical_columns, drop_first=True)

print("After encoding:", df_encoded.shape)
print(df_encoded.head())

#Day 2-3

# Separating features (X) and target (y)
X = df_encoded.drop(columns=['cnt', 'dteday'])  # cnt is the target column (total count of bikes rented)
y = df_encoded['cnt']

# Splitting the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Train shape:", X_train.shape)
print("Test shape:", X_test.shape)

# Linear Regression
lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)
y_pred_lin = lin_reg.predict(X_test)

# Random Forest
rf_reg = RandomForestRegressor(random_state=42)
rf_reg.fit(X_train, y_train)
y_pred_rf = rf_reg.predict(X_test)

# Evaluating Linear Regression
mae_lin = mean_absolute_error(y_test, y_pred_lin)
r2_lin = r2_score(y_test, y_pred_lin)

# Evaluating Random Forest
mae_rf = mean_absolute_error(y_test, y_pred_rf)
r2_rf = r2_score(y_test, y_pred_rf)

print("\nLinear Regression:")
print(f"MAE: {mae_lin:.2f}")
print(f"R²: {r2_lin:.2f}")

print("\nRandom Forest:")
print(f"MAE: {mae_rf:.2f}")
print(f"R²: {r2_rf:.2f}")

#day 4
import matplotlib.pyplot as plt

# --- Feature Importance for Random Forest ---
importances = rf_reg.feature_importances_
feature_names = X_train.columns

importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

print("\nTop 10 most important features:")
print(importance_df.head(10))

# --- Visualization: Feature Importance ---
plt.figure(figsize=(10, 6))
plt.barh(importance_df['Feature'][:10], importance_df['Importance'][:10])
plt.gca().invert_yaxis()
plt.title("Top 10 Important Features (Random Forest)")
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.show()

# --- Visualization: Actual vs Predicted ---
plt.figure(figsize=(6, 6))
plt.scatter(y_test, y_pred_rf, alpha=0.3, color='forestgreen')
plt.title("Random Forest: Actual vs Predicted")
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.show()

plt.figure(figsize=(6, 6))
plt.scatter(y_test, y_pred_lin, alpha=0.3, color='royalblue')
plt.title("Linear Regression: Actual vs Predicted")
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.show()

#day 5
results = pd.DataFrame({
    'Model': ['Linear Regression', 'Random Forest'],
    'MAE': [mae_lin, mae_rf],
    'R²': [r2_lin, r2_rf]
})

print("\nModel comparison:")
print(results)

print("Most important features (from Random Forest):")
print(importance_df.head(5))