# Write a program to Demonstrate the working of the Decision Tree Based ID3 Algorithm. Use an playtennis.csv data set for building the decision tree and apply this knowledge to classify a new sample.

import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
import matplotlib.pyplot as plt

# Load the dataset
filename = '/content/PlayTennis.csv'
try:
    data = pd.read_csv(filename)
    print("Dataset successfully loaded.")
except FileNotFoundError:
    print(f"Error: The file '{filename}' was not found.")
    exit()

# Display the first few rows of the dataset
print("Training Data:\n", data.head())

# Separate features (X) and target (y)
X = data.drop('play', axis=1)  # 'play' is the target column
y = data['play']

# Convert categorical variables to dummy/indicator variables
X = pd.get_dummies(X)

# Build the decision tree using the ID3 algorithm
clf = DecisionTreeClassifier(criterion='entropy')
clf = clf.fit(X, y)
# Visualize the decision tree
plt.figure(figsize=(12, 8))
tree.plot_tree(clf, feature_names=X.columns, class_names=clf.classes_, filled=True)
plt.title("Decision Tree Visualization")
plt.show()


# Ensure that the new sample contains all columns that were used during training
new_sample = pd.DataFrame({
    'outlook_sunny': [1],
    'outlook_overcast': [0],
    'outlook_rainy': [0],
    'temp_hot': [0],
    'temp_mild': [1],
    'temp_cool': [0],
    'humidity_high': [0],
    'humidity_normal': [1],
    'windy_False': [0],
    'windy_True': [1]
})

# Ensure the new sample has the same column order as the training data
new_sample = new_sample.reindex(columns=X.columns, fill_value=0)

# Predict the class for the new sample
prediction = clf.predict(new_sample)
print("Prediction for the new sample:", prediction[0])



