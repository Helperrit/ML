# Write a program to implement Logistic Regression considering iris dataset And perform the following operations on dataset 
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder

# Load the dataset
file_path = 'Iris.csv'  # Path to the uploaded Iris dataset
iris_df = pd.read_csv(file_path)

# Drop the 'Id' column as it is not relevant for modeling
iris_df.drop('Id', axis=1, inplace=True)

# Encode the 'Species' column to numerical values
label_encoder = LabelEncoder()
iris_df['Species'] = label_encoder.fit_transform(iris_df['Species'])

# Display the first 5 rows
print(iris_df.head())

# Check the number of samples of each class in species
print(iris_df['Species'].value_counts())

# Check for null values
print(iris_df.isnull().sum())

# Visualize the data
sns.pairplot(iris_df, hue='Species', palette='Set2')
plt.show()

# Compute covariance and correlation
print(iris_df.cov())
print(iris_df.corr())

# Prepare the data
X = iris_df.drop('Species', axis=1)
y = iris_df['Species']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Logistic Regression model
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Plot confusion matrix
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, cmap='coolwarm', fmt='d')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()
