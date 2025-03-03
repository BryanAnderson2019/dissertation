import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# Import necessary libraries
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

data = np.load("/Users/sankalpssss/Documents/marioenv/mario/mario2/Data final/Combined_dataset/Integer/Master_integer.npy",allow_pickle=True)

X = data[:, :-1]  # Features are all columns except the last one
y = data[:, -1]   # Target variable is the last column

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=42)  # 0.25 x 0.8 = 0.2

# Initialize the RFClassifier
model = RandomForestClassifier()

# Train the model
model.fit(X_train, y_train)

# Make predictions on validation data
val_predictions = model.predict(X_val)

# Evaluate the model on validation data
val_accuracy = accuracy_score(y_val, val_predictions)
print("Validation Accuracy:", val_accuracy)

# Make predictions on test data
test_predictions = model.predict(X_test)

# Evaluate the model on test data
test_accuracy = accuracy_score(y_test, test_predictions)
print("Test Accuracy:", test_accuracy)

import joblib

# Save the trained model
joblib.dump(model, '/Users/sankalpssss/Documents/marioenv/mario/mario2/Models/Final Models/RandomForest.pkl')
