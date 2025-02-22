# Multilayer Perceptron (MLP) classifier - Artificial Neural Network (ANN)

import numpy as np
import pandas as pd
import pickle as pk
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

# Read dataset/convert
dataset = pd.read_csv('phishing_dataset.csv')
dataset = dataset.to_numpy()

# Extract features/labels
features = dataset[:, :-1]
labels = dataset[:, -1]

print("Features shape:", features.shape)
print("Labels shape:", labels.shape)

# Print basic dataset information
print("\n1. Type: ", type(dataset))
print("\n2. Datapoints: ", features.shape[0])
print("\n3. Features: ", features.shape[1])
print("\n4. Class Labels Location: Final Column")
print("\n5. # Of Classes: ", len(np.unique(labels)))  # Use np.unique to count unique classes
# Normalize to ensure all values have a similar scale
# Larger features will disproportionately influence the algorithm
print("\n6. Pre-processing: Normalization")

# Normalize the features, scales them between 0 - 1.
fmin = np.min(features, axis=0)
fmax = np.max(features, axis=0)
features = (features - fmin) / (fmax - fmin + 1e-20) # Preventing non 0

# Split train and test features/labels
trn_f, tst_f, trn_l, tst_l = train_test_split(features, labels, test_size=0.2, random_state=42)  # Set random_state for reproducibility

# Train the model
model = MLPClassifier(hidden_layer_sizes=(20,20,10), verbose=True)
model.fit(trn_f, trn_l)

# Calculate training accuracy
trnAcc = model.score(trn_f, trn_l)
print("\nTraining Accuracy: ", 100 * trnAcc)

# Benchmark the model
print("Test Accuracy: ", 100 * accuracy_score(tst_l, model.predict(tst_f)))

# Calculate precision, recall, and F1 score
print("\nPrecision: ", 100 * precision_score(tst_l, model.predict(tst_f), average='weighted'))
print("Recall: ", 100 * recall_score(tst_l, model.predict(tst_f), average='weighted'))
print("F1 Score: ", 100 * f1_score(tst_l, model.predict(tst_f), average='weighted'))

# Print the classification report
print("\nClassification Report:")
print(classification_report(tst_l, model.predict(tst_f)))

# Save the trained model to disk
with open('trained_model.pkl', 'wb') as file:
    pk.dump(model, file)
    
print("\nTrained model saved to disk as 'trained_model.pkl'")
