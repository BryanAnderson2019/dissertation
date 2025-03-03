# Combining Sessions

The "[Combining_sessions](https://github.com/sankalp-s/MSC_Thesis/tree/main/Player_Inputs/Scripts/Combining_sessions)" folder contains scripts tailored for aggregating, preprocessing, and filtering datasets stored in .npy files, specifically curated for machine learning tasks within gaming environments.

## Scripts Overview

1. **[Combine_Image_sessions.py](https://github.com/sankalp-s/MSC_Thesis/blob/main/Player_Inputs/Scripts/Combining_sessions/Combine_Image_sessions.py)**: This script aggregates multiple datasets comprising image state and corresponding action data. It locates .npy files within a specified directory, loads the data, constructs a master dataset array, and saves it as "Master_image.npy". Summary information about the dataset's size and shape is provided.

2. **[Combine_RAM_sessions.py](https://github.com/sankalp-s/MSC_Thesis/blob/main/Player_Inputs/Scripts/Combining_sessions/Combine_RAM_sessions.py)**: Designed to convert binary arrays to integers and aggregate datasets stored in .npy files, this script deals with state and action data collected during interactions within gaming environments. Similar to the image sessions script, it locates .npy files, loads the data, performs necessary conversions, constructs a final aggregated dataset, and saves it as "Master_integer.npy". Summary information about the dataset's size and shape is also provided.

3. **[Preprocessing.py](https://github.com/sankalp-s/MSC_Thesis/blob/main/Player_Inputs/Scripts/Combining_sessions/Preprocessing.py)**: This script facilitates the preprocessing and filtering of datasets stored in .npy files. It encompasses functionalities such as loading data, extracting features and target variables, converting integers to binary arrays, mapping labels to indices, and filtering the dataset to remove instances solely characterized by forward movement. The script aims to enhance the dataset's suitability for multi-output multiclass classification tasks and ensures accurate model learning.

These scripts collectively provide essential functionalities for dataset aggregation, preprocessing, and filtering, making them invaluable tools for preparing datasets for machine-learning tasks.

# Gathering Inputs

The "[Gathering_Input](https://github.com/sankalp-s/MSC_Thesis/tree/main/Player_Inputs/Scripts/Gathering_Input)" folder comprises a collection of scripts designed to facilitate the extraction and preprocessing of input data for reinforcement learning tasks, particularly in the context of retro gaming environments.

## Scripts Overview

1. **[RAM_Input.py](https://github.com/sankalp-s/MSC_Thesis/blob/main/Player_Inputs/Scripts/Gathering_Input/RAM_Input.py)**: This script provides a framework for creating interactive gym environments, specifically tailored for retro games. It defines classes such as Interactive and RetroInteractive for environment initialization, state updates, rendering, user input handling, and main event loop execution. Each data point generated consists of an extracted RAM state (a 13x13 flattened array) and its corresponding action sequence, represented as a binary array of size 12.

2. **[Image_Input.py](https://github.com/sankalp-s/MSC_Thesis/blob/main/Player_Inputs/Scripts/Gathering_Input/Image_Input.py)**: Similar in structure to RAM_Input.py, this script fetches image states from the environment, processes them using wrappers (e.g., converting RGB images to grayscale, resizing), and generates a dataset comprising image-action pairs for training or analysis. The input states are represented as images (grayscale 84x84x1), while the action data remains consistent, represented as a binary array of size 12.

3. **[rominfo.py](https://github.com/sankalp-s/MSC_Thesis/blob/main/Player_Inputs/Scripts/Gathering_Input/rominfo.py)
**: This script provides functions to extract attributes from the RAM memory of the game Super Mario World. It includes functions to retrieve the agent's position, obtain information about sprites displayed on the screen, check for obstacles, and gather inputs within a specified radius around the agent. Additionally, the script contains a function to retrieve the entire RAM memory from the game environment, which can be useful for debugging or analysis purposes.

These scripts are intended to be incorporated into reinforcement learning pipelines where input data extraction and preprocessing are required. They provide foundational processes for handling input data, whether in the form of RAM states or image observations, and preparing datasets for training machine learning models.

# View or Replay Files

The "[View_or_Replay_Files](https://github.com/sankalp-s/MSC_Thesis/tree/main/Player_Inputs/Scripts/View_or_Replay_Files)" folder contains two Python scripts designed to facilitate the viewing and replaying of gameplay sessions stored in NumPy .npy files.

## Scripts Overview

1. **[replay_sessions.py](https://github.com/sankalp-s/MSC_Thesis/blob/main/Player_Inputs/Scripts/View_or_Replay_Files/replay_sessions.py)**: This script loads training data from a NumPy .npy file, simulates gameplay in the Super Mario World environment using the loaded actions, and renders the environment after each action. It utilizes the retro library to create the game environment and applies actions sequentially to simulate gameplay. Additionally, it provides an option to introduce a time delay to observe the game rendering more clearly.

2. **[view_files.py](https://github.com/sankalp-s/MSC_Thesis/blob/main/Player_Inputs/Scripts/View_or_Replay_Files/view_files.py)**: This script provides a function `open_npy_file()` to load data from a NumPy .npy file. Users can specify the file path, and the function returns the loaded data as a NumPy array. It includes error handling to manage cases where the file is not found or errors occur during loading.
