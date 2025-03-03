import retro
import os
import numpy as np
from rominfo import *
from joblib import load
import time

#Tensorflow models
import tensorflow as tf
from tensorflow import keras

# Load the saved model from the file
model = load('/Users/sankalpssss/Documents/marioenv/mario/mario2/Models/Final Models/RandomForest.pkl')

env = retro.make(game = "SuperMarioWorld-Snes",state='YoshiIsland2',obs_type = retro.Observations.IMAGE)

#Frame skip (hold an action for this many frames) and sticky action
obs = env.reset()

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
    return action_integer

while True:

    ram = getRam(env)
    state, x, y = getInputs(ram)
    state = np.reshape(state, (13, 13))

    predicted_action_integer = predict_action(state)

    """"
    act = integer_to_binary_array(predicted_action_integer[0])
    print("Action", "Array", predicted_action_integer, act)
    """

    if predicted_action_integer[0] == 0:
        act = np.array([0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0])
    else:
        act = integer_to_binary_array(predicted_action_integer[0])
    print("Action Integer", predicted_action_integer, "Action Array",  act)


    obs, rew, done, info = env.step(act)
    env.render()

    # Introduce a delay before processing the next state
    #time.sleep(delay_duration)
