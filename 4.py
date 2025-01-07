# Write a program to demonstrate the FIND-S algorithm for finding the most specific hypothesis based on a given set of training data samples. Read the training data from a enjoysport.CSV file.
import pandas as pd

def find_s_algorithm(data):
    # Initialize the most specific hypothesis with the first positive example
    hypothesis = ['0'] * (len(data.columns) - 1)
    
    for index, row in data.iterrows():
        if row.iloc[-1].lower() == 'yes':  # Ensure case-insensitivity
            if hypothesis == ['0'] * (len(data.columns) - 1):
                hypothesis = row.iloc[:-1].tolist()
            else:
                for i in range(len(hypothesis)):
                    if hypothesis[i] != row.iloc[i]:
                        hypothesis[i] = '?'
    
    return hypothesis

# Load the training data from the CSV file
filename = 'enjoysport.csv'  # Replace with the actual file path if needed
data = pd.read_csv(filename)

# Display the first 5 rows
print("Training Data:\n", data.head())

# Apply the FIND-S algorithm
specific_hypothesis = find_s_algorithm(data)

# Output the most specific hypothesis
print("\nMost Specific Hypothesis:", specific_hypothesis)
