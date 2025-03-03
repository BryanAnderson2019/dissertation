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
#model = load('/Users/sankalpssss/Documents/marioenv/mario/mario2/Models/Final Models/RandomForest.pkl')

model = tf.keras.models.load_model('/Users/sankalpssss/Documents/marioenv/mario/mario2/Models/Final Models/ANN_Filtered')


env = retro.make(game = "SuperMarioWorld-Snes",state='YoshiIsland2',obs_type = retro.Observations.IMAGE)

#Frame skip (hold an action for this many frames) and sticky action
obs = env.reset()

label_to_index = {
    0: 0, 8: 1, 16: 2, 24: 3, 32: 4, 40: 5, 48: 6, 56: 7,
    64: 8, 80: 9, 144: 10, 2048: 11, 2064: 12, 2072: 13,
    2080: 14, 2096: 15
}

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

    #labels to the index
    
    flattened_state = state.flatten()
    single_state  = (np.expand_dims(flattened_state,0))
    action_integer = model.predict(single_state)
    #return action_integer

    #for ANN
    index_single = np.argmax(action_integer)
    final_action_integer = index_to_label[index_single]
    return final_action_integer
    
# Define the state to be tested when no holding actions
test_state = np.array([
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
])

test_state2 = np.array([
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
])

test_state3 = np.array([
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
])

zero_state = np.array([
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
])
count = 0

#delay_duration = 0
blank_action = np.zeros(12)

G = True

while G:
    
    print("here7")
    ram = getRam(env)
    print("here5")
    state, x, y = getInputs(ram)
    print("here6")
    state = np.reshape(state, (13, 13))
    print("here2")
    
    if np.array_equal(state, test_state):
        predicted_action_integer =16
    else:
        count = count+1
        print(count)
        predicted_action_integer = predict_action(state)

    """"
    act = integer_to_binary_array(predicted_action_integer[0])
    print("Action", "Array", predicted_action_integer, act)
    """

    if predicted_action_integer == 0:
        act = np.array([0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0])
    else:
        act = integer_to_binary_array(predicted_action_integer)
    print("state",state, "Action Integer", predicted_action_integer, "Action Array",  act)
    print("here")

    obs, rew, done, info = env.step(act)
    obs, rew, done, info = env.step(blank_action)#Stop it from holding an action, like jump
    print("here3")
    env.render()
    print("here4")

    if np.array_equal(state, test_state):
        act = np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0])
        obs, rew, done, info = env.step(act)

    """
    if np.array_equal(state, test_state2):
        act = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0])
        obs, rew, done, info = env.step(act)

    if np.array_equal(state, test_state3):
        act = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0])
        obs, rew, done, info = env.step(act)
    
    """
        
    if np.array_equal(state, zero_state):
        act = np.array([1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0])
        obs, rew, done, info = env.step(act)
    
    # Introduce a delay before processing the next state
    #time.sleep(delay_duration)