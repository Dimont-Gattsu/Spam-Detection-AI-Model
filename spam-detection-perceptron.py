# Perceptron classifier - single-layer neural network

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import joblib

#    1. Reading the dataset into Python
#    2. Converting the data to NumPy
#    3. Extracting labels and features
#    4. Preprocessing the features and labels
#    5. Training the Perceptron model
#    6. Evaluating the model's performance
#    7. Saving the trained model to disk


# 1. Read dataset
dataset = pd.read_csv('spam.csv')

# 2. Convert data to NumPy
dataset = dataset.to_numpy()

# 3. Extract features and labels
features = dataset[:, :-1]
labels = dataset[:, -1]

# 4. Preprocess features and labels if required
# (No specific preprocessing steps mentioned in the lab)

# 5. Split the dataset into training and testing sets
trn_f, tst_f, trn_l, tst_l = train_test_split(features, labels, test_size=0.2, random_state=42)

# 6. Train the Perceptron model
model = Perceptron()
model.fit(trn_f, trn_l)

# 7. Evaluate the model's performance
trnAcc = accuracy_score(trn_l, model.predict(trn_f))
print("Training Accuracy: ", trnAcc)

tstAcc = accuracy_score(tst_l, model.predict(tst_f))
print("Test Accuracy: ", tstAcc)

# Additional evaluation metrics
print("\nPrecision: ", precision_score(tst_l, model.predict(tst_f), average='weighted'))
print("Recall: ", recall_score(tst_l, model.predict(tst_f), average='weighted'))
print("F1 Score: ", f1_score(tst_l, model.predict(tst_f), average='weighted'))

# Print classification report
print("\nClassification Report:")
print(classification_report(tst_l, model.predict(tst_f)))

# 8. Save the trained model to disk
joblib.dump(model, 'trained_perceptron_model.pkl')
print("\nTrained model saved to disk as 'trained_perceptron_model.pkl'")
