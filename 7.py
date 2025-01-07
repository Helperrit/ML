# Write a program to implement the Naïve Bayesian Classifier to classify the following English text.
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, recall_score, precision_score, confusion_matrix, classification_report

# Offline data
data = {
    'text': [
        'I love this sandwich', 'This is an amazing place', 'I feel very good about these cheese',
        'This is my best work', 'What an awesome view', 'I do not like this restaurant',
        'I am tired of this stuff', 'I can’t deal with this', 'He is my sworn enemy',
        'My boss is horrible', 'This is an awesome place', 'I do not like the taste of this juice',
        'I love to dance', 'I am sick and tired of this place', 'What a great holiday',
        'That is a bad locality to stay', 'We will have good fun tomorrow', 'I went to my enemys house today'
    ],
    'label': [
        'pos', 'pos', 'pos', 'pos', 'pos', 'neg', 'neg', 'neg', 'neg', 'neg',
        'pos', 'neg', 'pos', 'neg', 'pos', 'neg', 'pos', 'neg'
    ]
}

# Create a DataFrame from the offline data
df = pd.DataFrame(data)

# Display total instances
print("Total Instances of Dataset:", len(df))

# Preprocess the data
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df['text'])  # Convert text to feature vectors
y = df['label']  # Labels (positive or negative sentiment)

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Naïve Bayes classifier
nb = MultinomialNB()
nb.fit(X_train, y_train)

# Predict the labels for the test set
y_pred = nb.predict(X_test)

# Evaluate the classifier
accuracy = accuracy_score(y_test, y_pred)
recall = recall_score(y_test, y_pred, pos_label='pos', average='binary')
precision = precision_score(y_test, y_pred, pos_label='pos', average='binary')
conf_matrix = confusion_matrix(y_test, y_pred)

# Display evaluation results
print("\nAccuracy:", accuracy)
print("Recall:", recall)
print("Precision:", precision)
print("\nConfusion Matrix:\n", conf_matrix)
print("\nClassification Report:\n", classification_report(y_test, y_pred))
