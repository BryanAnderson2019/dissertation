# RAM-Action pair script

[RAM_Input.py](https://github.com/sankalp-s/MSC_Thesis/blob/main/Player_Inputs/Scripts/Gathering_Input/RAM_Input.py)

This script provides a framework for creating interactive gym environments, particularly designed for retro games. It leverages Pyglet for rendering the game environment and managing user input, enabling real-time interaction with the game.

Each data point consists of an extracted RAM state (a 13x13 flattened array) and its corresponding action sequence (presumably represented as a binary array of size 12).
### Overview

**Interactive Class:**  
The `Interactive` class serves as the base for interactive gym environments. It handles environment initialization, state updates, screen rendering, user input handling, and the main event loop.

**RetroInteractive Class:**  
The `RetroInteractive` class extends `Interactive` and is specialized for retro games. It facilitates the translation of keyboard inputs into gym actions suitable for retro game environments.

**Main Functionality:**  
The `main()` function orchestrates the script's operation. It parses command-line arguments to specify the retro game, state, scenario, and recording settings. Then, it initializes a `RetroInteractive` instance with these settings and runs the interactive game loop.

# Image-Action pair script

[Image_Input.py](https://github.com/sankalp-s/MSC_Thesis/blob/main/Player_Inputs/Scripts/Gathering_Input/Image_Input.py)

Each data point consists of an extracted RAM state (a 13x13 flattened array) and its corresponding action sequence (presumably represented as a binary array of size 12).

Note: that the script structure remains identical to the RAM-Action script, with the only distinction lying in the storage of data. In this version, the input states are represented as images, while the action data remains consistent, represented as a binary array of size 12.

### Overview

This script snippet demonstrates the process of fetching image states from an environment, processing them into a standardized format using wrappers, and generating a dataset comprising image-action pairs for training or analysis.

### Observation Retrieval

The script fetches observations (`obs`) from the environment. These observations typically represent the current state of the environment, usually in the form of images.

### Image Processing with Wrappers

Observations (`obs`) are processed through wrappers, which are preprocessing layers applied to the observations before further processing or training. Specifically, the wrappers convert RGB image observations into grayscale images with dimensions of 84x84x1. This process may involve operations such as converting RGB images to grayscale, resizing images to a standard size (84x84), and possibly other transformations for better compatibility with learning algorithms.

### Storing Image and Action Pair

After processing, the preprocessed images (grayscale 84x84x1) are stored as part of the state data (`self.states`). Additionally, actions taken in response to these observations are stored as part of the action data. These image-action pairs form the dataset used for training or analysis.

### Usage

This script snippet can be incorporated into reinforcement learning pipelines where it is necessary to preprocess image states and generate datasets for training machine learning models. It provides a foundational process for handling image observations in reinforcement learning environments and extracting meaningful data for training AI agents.

# ROM file

[rominfo.py](https://github.com/sankalp-s/MSC_Thesis/blob/main/Player_Inputs/Scripts/Gathering_Input/rominfo.py)

This script provides functions to extract attributes from the RAM memory of the game Super Mario World. It includes functions to retrieve the agent's position, obtain information about sprites displayed on the screen, check for obstacles, and gather inputs within a specified radius around the agent. These functions are useful for creating AI agents or analyzing game states in reinforcement learning tasks. Additionally, the script contains a function to retrieve the entire RAM memory from the game environment, which can be utilized for various debugging or analysis purposes.

