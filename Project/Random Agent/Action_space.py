import numpy as np

#Code obtained from https://www.timguelke.net/blog/2021/2/14/action-space-for-the-openai-retro-gym-game-airstriker-genesis
# Define button names and corresponding actions for the game
buttons = ["B", "Y", "SELECT", "START", "UP", "DOWN", "LEFT", "RIGHT", "A", "X", "L", "R"]
actions = [["B"], ["Y"], ["SELECT"], ["START"], ["UP"], ["DOWN"], ["LEFT"], ["RIGHT"], ["A"], ["X"], ["L"], ["R"]]
actions_ag = []

# Convert button actions to binary arrays
for action in actions:
    arr = np.array([0] * 12)
    for button in action:
        arr[buttons.index(button)] = 1
    actions_ag.append(arr)

# Print and label the binary action arrays
i = 0
while i < 11:
    act = actions_ag[i]
    print(act)
    i=i+1
    print(act.shape)