# Replay Sessions

[replay_sessions.py](https://github.com/sankalp-s/MSC_Thesis/blob/main/Player_Inputs/Scripts/View_or_Replay_Files/replay_sessions.py)

This Python script loads training data from a NumPy .npy file, simulates gameplay in the Super Mario World environment using the loaded actions, and renders the environment after each action. It leverages the retro library to create the game environment and applies actions sequentially to simulate gameplay. Optionally, a time delay can be introduced to observe the game rendering more clearly.

# View files

[view_files.py](https://github.com/sankalp-s/MSC_Thesis/blob/main/Player_Inputs/Scripts/View_or_Replay_Files/view_files.py)

This Python script provides a function `open_npy_file()` to load data from a NumPy `.npy` file. It allows users to specify the file path and returns the loaded data as a NumPy array. The function includes error handling to deal with cases where the file is not found or errors occur during loading.

## Summary of Functionality:

### `open_npy_file(file_path):`
- Opens a `.npy` file located at the specified `file_path`.
- Returns the loaded data as a NumPy array if successful.
- Handles errors gracefully, printing appropriate messages for file not found or other exceptions.



