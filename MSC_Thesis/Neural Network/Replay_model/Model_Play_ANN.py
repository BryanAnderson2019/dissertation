print("code start")
import retro
import sys
import os
import numpy as np
sys.path.append(os.path.abspath("/home/bryan/MSC_Thesis/Player_Inputs/Scripts/Gathering_Input"))
from rominfo import *
from joblib import load
import time

print("everything has been imported correctly 1")
#Tensorflow models
import tensorflow as tf
print("tensorflow has been imported correctly")
from tensorflow import keras

print("everything has been imported correctly 3")
# Load the saved model from the file
model = tf.keras.models.load_model('/home/bryan/MSC_Thesis/Neural Network/Trained_ANN_Models/ANN_Final_Preprocessed')


env = retro.make(game = "SuperMarioWorld-Snes",state='YoshiIsland2',obs_type = retro.Observations.IMAGE)

#Frame skip (hold an action for this many frames) and sticky action
obs = env.reset()

#change the custom labels everytime according to the dataset used.
label_to_index = {8: 0, 24: 1, 64: 2, 144: 3, 2048: 4, 2064: 5, 2072: 6}

# Create a mapping from class indices to original labels which will be later used to track the labels from the class index.
index_to_label = {index: label for label, index in label_to_index.items()}

def integer_to_binary_array(integer_value, array_length=12):
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
    return np.array(binary_array)

def predict_action(state):
    
    flattened_state = state.flatten()
    single_state  = (np.expand_dims(flattened_state,0))
    action_integer = model.predict(single_state)

    #for ANN
    index_single = np.argmax(action_integer)
    final_action_integer = index_to_label[index_single]
    return final_action_integer

#delay_duration = 0
blank_action = np.zeros(12)

G = True

while G:
    
    ram = getRam(env)
    state, x, y = getInputs(ram)
    state = np.reshape(state, (13, 13))
    
    predicted_action_integer = predict_action(state)


    act = integer_to_binary_array(predicted_action_integer)
    print("state",state, "Action Integer", predicted_action_integer, "Action Array",  act)

    obs, rew, done, info = env.step(act)
    obs, rew, done, info = env.step(blank_action)#Stop it from holding an action, like jump
    env.render()
