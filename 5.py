# Write a program to demonstrate the Candidate-Elimination algorithm to output a description of the set of all hypotheses consistent with the training examples. Read the training data from a playtennis.CSV file.

import pandas as pd
def candidate_elimination_algorithm(data):
    # Initialize specific and general hypotheses
    num_attributes = len(data.columns) - 1
    specific_hypothesis = ['0'] * num_attributes
    general_hypothesis = [['?'] * num_attributes]
    
    # Iterate through each row in the dataset
    for index, row in data.iterrows():
        # Get the target value ('Yes' or 'No')
        target = row[-1]
        
        if target == 'Yes':  # Positive example
            for i in range(num_attributes):
                if specific_hypothesis[i] == '0':  # If it's the first positive example
                    specific_hypothesis[i] = row[i]
                elif specific_hypothesis[i] != row[i]:  # If there's a conflict
                    specific_hypothesis[i] = '?'
            
            # Generalize the hypotheses that are not consistent with the current example
            general_hypothesis = [h for h in general_hypothesis if all(h[i] == '?' or h[i] == row[i] for i in range(num_attributes))]
        
        elif target == 'No':  # Negative example
            new_general_hypothesis = []
            for h in general_hypothesis:
                # For each general hypothesis, create a new hypothesis for each attribute
                for i in range(num_attributes):
                    if h[i] == '?':
                        if specific_hypothesis[i] != row[i]:
                            new_hypothesis = h[:]
                            new_hypothesis[i] = specific_hypothesis[i]
                            if new_hypothesis not in new_general_hypothesis:
                                new_general_hypothesis.append(new_hypothesis)
            general_hypothesis = new_general_hypothesis
    
    return specific_hypothesis, general_hypothesis

# Load the training data from CSV
filename = '/content/PlayTennis.csv'
data = pd.read_csv(filename)

# Display the first 5 rows
print("Training Data:\n", data.head())

# Apply the Candidate-Elimination algorithm
specific_hypothesis, general_hypothesis = candidate_elimination_algorithm(data)

# Output the most specific and general hypotheses
print("\nSpecific Hypothesis:", specific_hypothesis)
print("\nGeneral Hypothesis:")
for h in general_hypothesis:
    print(h)
