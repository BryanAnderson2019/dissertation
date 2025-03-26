import sys
import os
import numpy as np
sys.path.append(os.path.abspath("/home/bryan/MSC_Thesis/Player_Inputs/Scripts/Gathering_Input"))
from rominfo import *
import retro
import time

buttons = ["B", "Y", "SELECT","START", "UP", "DOWN", "LEFT", "RIGHT", "A", "X", "L", "R"]
SaveLocationLosing = "/home/bryan/dissertation/MSC_Thesis/Player_Inputs/Dataset/lossing/YoshiIsland2/"
SaveLocationWinning = "/home/bryan/dissertation/MSC_Thesis/Player_Inputs/Dataset/winning/YoshiIsland2/"
fileName = "test"

save = True
arrStart = 0
arrEnd = 0
loses = 0

# Load the Super Mario World environment
env = retro.make(game="SuperMarioWorld-Snes", state='YoshiIsland2', obs_type=retro.Observations.IMAGE)

# Load training data from the .npy file
Training = np.load("/home/bryan/MSC_Thesis/Player_Inputs/Dataset/YoshiIsland1/TanushreeL1.npy", allow_pickle=True)

Xtrain = Training[0]
Ytrain = Training[1]
data = Training[1]
print(Ytrain.shape)  # Print the shape of the training labels array

# Fix issues with numpy loading
Ytrain = np.array([np.array(val) for val in Ytrain])
Ytrain = np.array([val.reshape(1, 12) for val in Ytrain])  # Reshape to fit model

print(Ytrain.shape)  # Print the shape of the training labels array

# Reset the environment
obs = env.reset()

# Iterate over actions and render the environment
for action in Ytrain:
    arrEnd += 1

    for a in action:     
        obs, rew, done, _info = env.step(a)  # Use the generated action
        #print(f"rew = {rew}")
        #print(f"done = {done}")
        ram = getRam(env)
        marioX, marioY, layer1x, layer1y  = getXY(ram)
        # print(f"marioX = {marioX}, marioY = {marioY}")

        if (marioY == 0 and save):
            loses += 1
            np.save((SaveLocationLosing + fileName + str(loses)),np.array((Xtrain[arrStart:arrEnd],data[arrStart:arrEnd])))
            print(f"loses = {loses}")
            print(f"arrStart = {arrStart}")
            print(f"arrEnd = {arrEnd}")
            save = False
        elif (marioX > 16 and save != True):
            save = True
            print(f"save = {save}")
            arrStart = arrEnd
            print(f"arrEnd = {arrEnd}")
            print(f"arrStart now equals = {arrStart}")
        elif (rew == 100 and save):
            np.save((SaveLocationWinning + fileName),np.array((Xtrain[arrStart:arrEnd],data[arrStart:arrEnd])))
            print(f"mario finished the level")
            print(f"loses = {loses}")
            print(f"arrStart = {arrStart}")
            print(f"arrEnd = {arrEnd}")
            save = False
        
        env.render()  # Render the environment
        # Optionally, introduce a time delay to observe the game rendering
        # time.sleep(0.05)
