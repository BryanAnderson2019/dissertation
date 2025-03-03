import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import retro
import time

# Import necessary libraries
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

from joblib import load

# Load the saved model from the file
model = load('/Users/sankalpssss/Documents/marioenv/mario/mario2/Models/RF_Jumping.pkl')

data = np.load("/Users/sankalpssss/Documents/marioenv/mario/mario2/Data final/Just_States/State2_integer.npy",allow_pickle=True)

X = data[:, :-1]  # Features are all columns except the last one
y = data[:, -1]   # Target variable is the last column

y_pred = model.predict(X)

print('----------------------------')
print(data.shape)
print(y_pred.shape)
print(y_pred)
print('----------------------------')

def integer_to_binary_array(integer_value, array_length):
    """
    Convert an integer to a binary array of specified length.
    Parameters:
        integer_value (int): The integer value to be converted.
        array_length (int): The desired length of the binary array.
    Returns:
        list: The binary array representing the integer value.
    """
    binary_string = format(integer_value, 'b')  # Convert integer to binary string
    binary_array = [int(bit) for bit in binary_string.zfill(array_length)]  # Pad with leading zeros if needed
    return binary_array

def convert_integers_to_binary_arrays(data):
    """
    Convert integer values in the dataset back to binary arrays.
    Parameters:
        data (numpy.ndarray): The dataset containing integer values.
    Returns:
        numpy.ndarray: The dataset with integer values converted back to binary arrays.
    """
    binary_arrays_list = []
    for integer in data:
        binary_array = integer_to_binary_array(integer, array_length=12)  # Adjust array_length as needed
        binary_arrays_list.append(binary_array)
    return np.array(binary_arrays_list)

# Convert integer values back to binary arrays
binary_data = convert_integers_to_binary_arrays(y_pred)


print("-----------ACTIONS---------------")
print(binary_data[0])

#Replay Inputs

# Load the Super Mario World environment
env = retro.make(game="SuperMarioWorld-Snes", state='YoshiIsland2', obs_type=retro.Observations.IMAGE)

# Reset the environment
obs = env.reset()

# Iterate over actions and render the environment
for action in binary_data:
    obs, rew, done, _info = env.step(action)  # Use the generated action
    env.render()  # Render the environment
    # Optionally, introduce a time delay to observe the game rendering
    # time.sleep(1)
