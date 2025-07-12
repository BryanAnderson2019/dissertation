import sys
import os
import numpy as np
sys.path.append(os.path.abspath("/home/bryan/MSC_Thesis/Player_Inputs/Scripts/Gathering_Input"))
from rominfo import *
import retro
import time

buttons = ["B", "Y", "SELECT","START", "UP", "DOWN", "LEFT", "RIGHT", "A", "X", "L", "R"]
# Load the Super Mario World environment
env = retro.make(game="SuperMarioWorld-Snes", state='YoshiIsland2', obs_type=retro.Observations.IMAGE)

# Load training data from the .npy file
Training = np.load("/home/bryan/dissertation/best_run.npy", allow_pickle=True)

arr = [0, 1, 0, 1, 1, 1, 0, 0, 1, 0, 1, 1]
result = 0
for i in range(12):
    result += arr[i]*pow(2, 12-(i+1))
print(result)

test = np.load("/home/bryan/dissertation/best_run.npy", allow_pickle=True)
print(test.shape)
Xtest = np.array([val[:169] for val in test])
Ytest = np.array([val[169:] for val in test])
print(Xtest.shape)
print(Ytest.shape)

Xtrain = Training[0]
Ytrain = Training[1]
print(Ytrain)  # Print the shape of the training labels array
print(Ytrain.shape) 

# Fix issues with numpy loading
Ytrain = np.array([np.array(val) for val in Ytrain])
Ytrain = np.array([val.reshape(1, 12) for val in Ytrain])  # Reshape to fit model
print(Ytrain.shape)  # Print the shape of the training labels array

# Reset the environment
obs = env.reset()

# Iterate over actions and render the environment
for action in Ytrain:
    for a in action:
        b = []
        for index in range(12):
            if a[index] == 1:
                b.append(buttons[index])

        #print(f"a = {a}")
        #print(f"buttons = {b}")

        obs, rew, done, _info = env.step(a)  # Use the generated action
        ram = getRam(env)
        marioX, marioY, layer1x, layer1y  = getXY(ram)

        #print(f"done = {done}")

        if(rew == 100):
            print(f"rew = {rew}")
            print(f"marioX = {marioX}, marioY = {marioY}")

        print(f"marioX = {marioX}, marioY = {marioY}")

        #print(f"marioX = {marioX}, marioY = {marioY}")
        
        #if (marioX < 4820):
        #    break
        #else:
        #    time.sleep(0.05)
        #    print(f"marioX = {marioX}, marioY = {marioY}")

        #if (rew == 100):
        #    quit()
        
        env.render()  # Render the environment
        # Optionally, introduce a time delay to observe the game rendering
        #time.sleep(0.05)
