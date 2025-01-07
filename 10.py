# Write a program to Demonstrate RANDOM FOREST using the dataset California housing price prediction and perform the following operations on dataset 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Load dataset
data = pd.read_csv('california_housing.csv')

# Display basic info and first 5 rows
print("Columns:", data.columns)
print("\nFirst 5 Rows:\n", data.head())

# Distribution of target and null values
print("\nTarget Distribution:\n", data['median_house_value'].value_counts(bins=10))
print("\nNull Values:\n", data.isnull().sum())

# Visualization
plt.figure(figsize=(10, 6))
sns.histplot(data['median_house_value'], bins=30, kde=True, color='blue')
plt.title('Target Distribution')
plt.show()

# Correlation matrix with only numeric data
numeric_data = data.select_dtypes(include=[np.number])

plt.figure(figsize=(12, 8))
sns.heatmap(numeric_data.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title('Correlation Matrix')
plt.show()

# One-hot encoding and split
X = pd.get_dummies(data.drop(columns='median_house_value'))
y = data['median_house_value']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train and predict with Random Forest
regressor = RandomForestRegressor(n_estimators=100, random_state=42)
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)

# Evaluation
print("\nMSE:", mean_squared_error(y_test, y_pred))
print("R²:", r2_score(y_test, y_pred))

plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.6, color='orange')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
plt.title('Actual vs Predicted')
plt.show()
