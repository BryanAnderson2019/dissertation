Notebooks:

# Random Agent

[Randominput_Agent.py](https://github.com/sankalp-s/MSC_Thesis/blob/main/Random%20Agent/Randominput_Agent.py)

The Python script interacts with the Super Mario World environment in Gym Retro, an environment for training reinforcement learning agents. It takes random actions within the game and updates key press counts and total rewards based on the agent's actions. Once the episode is completed, it prints the key press counts and generates plots to visualize the total rewards against time and the key press counts as a bar graph.

Video of the Random agent performing can be found here: [LINK](https://www.youtube.com/watch?v=l4LiI27aR5g)




# Action Space Mapping

[Action_space.py](https://github.com/sankalp-s/MSC_Thesis/blob/main/Random%20Agent/Action_space.py)

This code snippet defines a mapping between human-readable button names and binary action arrays for the OpenAI Retro Gym games. Each action corresponds to a combination of button presses in the game.

## Explanation

- The `buttons` list contains the names of the buttons available in the game.
- The `actions` list contains lists of buttons corresponding to each action. Each sublist represents a unique action in the game.
- The loop iterates over each action in `actions` and creates a binary array representing the button presses for that action. The array has a length of 12, where each index corresponds to a button in the `buttons` list. If a button is part of the action, its corresponding index in the array is set to 1; otherwise, it remains 0.
- The resulting binary arrays are stored in the `actions_ag` list.
