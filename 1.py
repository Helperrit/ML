# 1. Write a program to implement Linear Regression Algorithm using the dataset boston housing price prediction And perform the following operations on dataset 
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load the dataset
file_path = '/content/HousingData.csv'
boston_df = pd.read_csv(file_path)

# Handle missing values (drop rows with missing target or impute missing values)
boston_df = boston_df.dropna(subset=['MEDV'])  # Drop rows where target is missing
boston_df = boston_df.fillna(boston_df.median())  # Impute other missing values with median

# Check the first few rows and for missing values
print(boston_df.head())
print(boston_df.isnull().sum())

# Visualize selected features
sns.pairplot(boston_df[['MEDV', 'RM', 'LSTAT', 'PTRATIO']])
plt.show()

# Covariance and correlation
print(boston_df.cov())
print(boston_df.corr())

# Prepare data for modeling
X = boston_df.drop('MEDV', axis=1)
y = boston_df['MEDV']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a linear regression model
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Evaluate the model
print('Mean Squared Error:', mean_squared_error(y_test, y_pred))
print('R^2 Score:', r2_score(y_test, y_pred))

# Scatter plot of actual vs predicted prices
plt.scatter(y_test, y_pred)
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.title('Actual vs Predicted Prices')
plt.show()
