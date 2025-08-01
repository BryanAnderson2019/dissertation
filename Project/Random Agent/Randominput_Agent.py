# Import necessary libraries
import retro
import numpy as np
import matplotlib.pyplot as plt
import time

# Create Retro environment for Super Mario World
env = retro.make(game='SuperMarioWorld-Snes')
env.reset()

# Initialize variables to track key press counts
B_Value = 0
Y_Value = 0
SELECT_Value = 0
START_Value = 0
UP_Value = 0
DOWN_Value = 0
LEFT_Value = 0
RIGHT_Value = 0
A_Value = 0
X_Value = 0
L_Value = 0
R_Value = 0

# Initialize variables to track rewards and time
reward_total = 0
reward_array = []
reward_count_array = []

# Array to store key press counts for each key
key_press_counts = np.zeros(12)

# Start time for tracking duration
start = time.time()

# Main loop to interact with the Retro environment
while True:
    # Take a random action in the environment
    action = env.action_space.sample()
    obs, rew, done, info = env.step(action)
    env.render()
    
    # Update rewards if positive
    if rew >= 0:
        reward_total += rew
        reward_array.append(reward_total)
        reward_count_array.append(time.time() - start)
        #print(time.time())  # For debugging purposes
                
    # Update key press counts
    for i, pressed in enumerate(action):
        if pressed:
            # Increment count for the pressed key
            key_press_counts[i] += 1
    
    # Check if the episode is done
    if done:
        # Print key press counts
        print("Key Press Counts:")
        print(f"B: {B_Value}, Y: {Y_Value}, SELECT: {SELECT_Value}, START: {START_Value}, UP: {UP_Value}, DOWN: {DOWN_Value}, LEFT: {LEFT_Value}, RIGHT: {RIGHT_Value}, A: {A_Value}, X: {X_Value}, L: {L_Value}, R: {R_Value}")
        
        # Plot rewards against time
        plt.plot(reward_count_array, reward_array)
        plt.xlabel("Time (s)")
        plt.ylabel("Total Rewards")
        plt.title("Total Rewards vs. Time")
        plt.grid(True)
        plt.show()

        # Plot key press counts as a bar graph
        keys = ['B', 'Y', 'SELECT', 'START', 'UP', 'DOWN', 'LEFT', 'RIGHT', 'A', 'X', 'L', 'R']
        plt.bar(keys, key_press_counts)
        plt.xlabel("Keys")
        plt.ylabel("Key Press Counts")
        plt.title("Key Press Counts")
        plt.show()
        
        # End the loop
        break

