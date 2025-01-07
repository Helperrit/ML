# Write a program to Demonstrate the working of the KNN. Use an IRIS  data set 
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load the CSV dataset (replace with the actual path to your CSV file)
file_path = '/content/Iris.csv'  # Replace with the path to your CSV file
data = pd.read_csv(file_path)

# Display the first 5 rows of the dataset
print("Dataset Head:\n", data.head())

# Assuming the dataset has features in all columns except the last one as 'target'
# Split the data into features (X) and target (y)
X = data.iloc[:, :-1]  # Select all columns except the last one as features
y = data.iloc[:, -1]   # Select the last column as target

# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the KNN classifier with k=3
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

# Make predictions on the test data
y_pred = knn.predict(X_test)

# Evaluate the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
print("\nAccuracy of the KNN model:", accuracy)

# Detailed classification report
print("\nClassification Report:\n", classification_report(y_test, y_pred))