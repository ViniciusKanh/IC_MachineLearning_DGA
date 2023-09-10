import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import ADASYN
from opfython.models import SupervisedOPF
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, confusion_matrix
from tabulate import tabulate

# Load the dataset
input_file = '0-Datasets/Dataset_Tratado.data'
names = ['Defeito', 'H2', 'CH4', 'C2H2', 'C2H4', 'C2H6']
df = pd.read_csv(input_file, names=names)

# Select the features and the target
features = ['H2', 'CH4', 'C2H2', 'C2H4', 'C2H6']
target = 'Defeito'
X = df[features].values
y = df[target].values

# Normalize the features using min-max scaling
scaler = MinMaxScaler()
X = scaler.fit_transform(X)

# Make sure y labels are in correct format for OPFython
unique_y = sorted(set(y))
label_map = {label: i for i, label in enumerate(unique_y)}
y = np.array([label_map[label] for label in y])

# Create a KFold object
kf = KFold(n_splits=10, shuffle=True, random_state=42)

# Initialize lists to hold scores for each fold
accuracy_scores = []
precision_scores = []
recall_scores = []
f1_scores = []

# Perform 10-fold cross validation
for train_index, test_index in kf.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    
    # Train the Optimum-Path Forest classifier
    opf = SupervisedOPF(distance='euclidean')
    opf.fit(X_train, y_train)  # Use the training data

    # Test the Optimum-Path Forest classifier
    y_pred = opf.predict(X_test)

    # Calculate the metrics
    accuracy_scores.append(accuracy_score(y_test, y_pred))
    precision_scores.append(precision_score(y_test, y_pred, average='macro'))
    recall_scores.append(recall_score(y_test, y_pred, average='macro'))
    f1_scores.append(f1_score(y_test, y_pred, average='macro'))
    
# Calculate the average scores
avg_accuracy = np.mean(accuracy_scores)
avg_precision = np.mean(precision_scores)
avg_recall = np.mean(recall_scores)
avg_f1 = np.mean(f1_scores)

# Print the average scores
print(f'Média da Acurácia: {avg_accuracy:.2f}')
print(f'Média da Precisão: {avg_precision:.2f}')
print(f'Média da Revocação (Recall): {avg_recall:.2f}')
print(f'Média do F1 Score: {avg_f1:.2f}')
