import numpy as np
import pandas as pd

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