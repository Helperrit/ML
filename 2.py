# Write a program to Demonstrate Gradient Descent algorithm using the dataset California housing price prediction and perform the following operations on dataset 
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Load the dataset
file_path = 'housing.csv'  # Update with the correct file path if necessary
california_df = pd.read_csv(file_path)

# Rename target column for consistency
california_df.rename(columns={'median_house_value': 'PRICE'}, inplace=True)

# Handle missing values in 'total_bedrooms' by imputing the median
california_df['total_bedrooms'].fillna(california_df['total_bedrooms'].median(), inplace=True)

# Drop the 'ocean_proximity' column as it's categorical and not suitable for regression
california_df.drop('ocean_proximity', axis=1, inplace=True)

# Display the first 5 rows
print(california_df.head())

# Check for null values
print(california_df.isnull().sum())

# Visualize the data
sns.pairplot(california_df[['PRICE', 'median_income', 'housing_median_age', 'total_rooms']])
plt.show()

# Compute covariance and correlation
print(california_df.cov())
print(california_df.corr())

# Prepare the data
X = california_df.drop('PRICE', axis=1)
y = california_df['PRICE']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train using Gradient Descent
sgd = SGDRegressor(max_iter=1000, tol=1e-3, random_state=42)
sgd.fit(X_train_scaled, y_train)

# Predict and evaluate
y_pred = sgd.predict(X_test_scaled)
print('Mean Squared Error:', mean_squared_error(y_test, y_pred))
print('R^2 Score:', r2_score(y_test, y_pred))

# Scatter plot of actual vs predicted prices
plt.scatter(y_test, y_pred)
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.title('Actual vs Predicted Prices')
plt.show()
