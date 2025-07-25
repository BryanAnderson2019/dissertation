import sys
import os
import numpy as np
sys.path.append(os.path.abspath("/home/bryan/MSC_Thesis/Player_Inputs/Scripts/Gathering_Input"))
from rominfo import *
import retro
import time
import gzip

#Code obtained from https://www.timguelke.net/blog/2021/2/14/action-space-for-the-openai-retro-gym-game-airstriker-genesis
# Define button names and corresponding actions for the game
buttons = ["B", "Y", "SELECT", "START", "UP", "DOWN", "LEFT", "RIGHT", "A", "X", "L", "R"]
actionsArray = [["B"], ["Y"], ["SELECT"], ["START"], ["UP"], ["DOWN"], ["LEFT"], ["RIGHT"], ["A"], ["X"], ["L"], ["R"]]
actions_ag = []
played_actions = []
index = 1
view = True

env = retro.make(game="SuperMarioWorld-Snes", state='YoshiIsland2', scenario=None, obs_type=retro.Observations.IMAGE)
obs = env.reset()
env.em.get_state()

# Convert button actions to binary arrays
for action in actionsArray:
    arr = np.array([0] * 12)
    for button in action:
        arr[buttons.index(button)] = 1
    actions_ag.append(arr)

# Print and label the binary action arrays
for i in range(12):
    act = actions_ag[i]
    print(act)
    print(act.shape)

def playActions(actions, times):
    global index
    global view
    # convert the actions into a 12-bit binary number 
    arr = np.array([0] * 12)
    for button in actions:
        arr[buttons.index(button)] = 1

    # playes the actions for the amount of times specified
    for x in range(times):
        obs, rew, done, _info = env.step(arr)  # Use the generated action
        print("rendering")
        env.render()  # Render the environment

        # prints out the action on the terminal
        print(f"action {index}")
        print(f"{arr}")
        print(f"button(s): {actions} \n")
        index += 1
        # adds a ime delay
        #time.sleep(1)

def save_state_to_file(env, name="test.state"):
    
    content = env.em.get_state()
    with gzip.open(name, 'wb') as f:
        f.write(content)

    ram = getRam(env)
    marioX, marioY, layer1x, layer1y  = getXY(ram)
    print(f"marioX = {marioX}, marioY = {marioY}")

playActions(["Y", "RIGHT"], 232)
playActions(["Y", "RIGHT"], 206)
playActions(["RIGHT", "A"], 129)
playActions(["Y", "RIGHT"], 129)
playActions(["B"], 5)
playActions(["RIGHT", "A"], 18)
playActions(["B"], 146)
playActions(["RIGHT", "A"], 146)
playActions(["X"], 146)
playActions(["B", "Y", "RIGHT"], 97)
playActions(["RIGHT", "A"], 193)
playActions(["RIGHT", "A"], 132)
playActions(["B"], 38)
playActions(["B"], 183)

playActions(["Y", "RIGHT"], 93)
playActions(["RIGHT", "A"], 93)
playActions(["Y", "RIGHT"], 93)
playActions(["B"], 93)
playActions(["RIGHT", "A"], 93)
playActions(["X"], 93)
playActions(["B", "Y", "RIGHT"], 93)
playActions(["RIGHT", "A"], 93)
playActions(["B"], 93)
playActions(["X"], 93)
playActions(["B", "LEFT"], 93)
playActions(["Y", "LEFT"], 93)


#save_state_to_file(env) # use this to save the state of the game

