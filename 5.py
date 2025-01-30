# Write a program to demonstrate the Candidate-Elimination algorithm to output a description of the set of all hypotheses consistent with the training examples. Read the training data from a playtennis.CSV file.

import pandas as pd

# Load dataset
data = pd.read_csv('playtennis.csv') 
X = data.iloc[:, :-1].values  # Features
Y = data.iloc[:, -1].values   # Target

# Initialize specific and general hypotheses
s_hypo = ['?'] * len(X[0])  # Most specific hypothesis
g_hypo = [['?'] * len(X[0])]  # Most general hypothesis

# Candidate-Elimination Algorithm
for i in range(len(X)):
    if Y[i] == 'Yes':  # Positive example
        for j in range(len(X[i])):
            if s_hypo[j] == '?':
                s_hypo[j] = X[i][j]
            elif s_hypo[j] != X[i][j]:
                s_hypo[j] = '?'
    else:  # Negative example
        for j in range(len(X[i])):
            if s_hypo[j] != '?' and s_hypo[j] != X[i][j]:
                g_hypo.append(['?' if k != j else s_hypo[k] for k in range(len(X[i]))])

print("Specific Hypothesis:", s_hypo)
print("General Hypotheses:", g_hypo)
