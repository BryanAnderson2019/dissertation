# Combine Image sessions

[Combine_Image_sessions.py](https://github.com/sankalp-s/MSC_Thesis/blob/main/Player_Inputs/Scripts/Combining_sessions/Combine_Image_sessions.py)

This script is designed to aggregate multiple datasets stored in `.npy` files, each containing image state and corresponding action data. 

## Functionality

The script performs the following operations:

1. **Locating and Listing `.npy` Files**: It locates all `.npy` files within a specified directory containing the datasets.

2. **Loading and Aggregating Data**: For each `.npy` file found, the script loads the data and appends the image states and actions into separate lists.

3. **Constructing Master Dataset**: After aggregating all datasets, the script constructs two arrays: `final_state_array` containing the aggregated image-state data, and `final_action_array` containing the corresponding action data.

4. **Concatenating and Saving Dataset**: The image-state and action arrays are concatenated to form a master dataset array named `dataset`, which is then saved as a `.npy` file for future use.

5. **Outputting Summary Information**: The script provides summary information regarding the size and shape of the combined dataset, offering insights into the dimensions of the image-state and action data arrays.

## Usage

To use the script:
- Ensure that the directory containing the `.npy` files is properly specified.
- Run the script to aggregate the datasets and generate a master dataset file.

## Output

Upon execution, the script generates a master dataset file named `Master_image.npy`, containing the combined image-state and action data. Additionally, summary information regarding the size and shape of the dataset is displayed for reference.


# Combine RAM sessions

[Combine_RAM_sessions.py](https://github.com/sankalp-s/MSC_Thesis/blob/main/Player_Inputs/Scripts/Combining_sessions/Combine_RAM_sessions.py)

This script is designed to convert binary arrays to integers and aggregate multiple datasets stored in `.npy` files. Each dataset comprises state and action data collected during interactions with the environment.

## Functionality

The script performs the following operations:

1. **Locating and Listing `.npy` Files**: It locates all `.npy` files within a specified directory containing the datasets.

2. **Loading Data**: For each `.npy` file found, the script loads the data using a custom function `open_npy_file`.

3. **Converting Binary Arrays to Integers**: The script includes a function `binary_array_to_integer` to convert binary arrays to integers.

4. **Aggregating Data**: After loading each dataset, the script converts binary action arrays to integers and appends them to the corresponding state arrays. These appended arrays are collected in a list.

5. **Final Dataset Construction**: Once all datasets are processed, the script vertically stacks the collected session data to create a final aggregated dataset.

6. **Outputting Summary Information**: The script displays the resulting shape of each processed dataset and the shape of the final aggregated dataset.

## Usage

To use the script:
- Ensure that the directory containing the `.npy` files is properly specified.
- Run the script to convert binary arrays to integers, aggregate the datasets, and generate a final dataset.

## Output

Upon execution, the script generates a master dataset file named `Master_integer.npy`, comprising 120,224 samples. Each sample contains 170 features. The first 169 elements (indices 0 to 168) represent input features, specifically the RAM states of the game. These values encode various aspects of the game environment at a specific time, including player position, enemy locations, and other relevant game state information. The last element (index 169) represents the target variable, indicating the action taken by the player in response to the observed game state. This setup conforms to the standard format for training machine learning models, where input features are utilized to predict or classify the target variable.


# Preprocessing

[Preprocessing.py](https://github.com/sankalp-s/MSC_Thesis/blob/main/Player_Inputs/Scripts/Combining_sessions/Preprocessing.py)

This script facilitates preprocessing and filtering of a dataset stored in a .npy file. It encompasses several key functionalities to optimize the dataset for multi-output multiclass classification tasks.

## Functionality

The script performs the following operations:

1. **Loading Data**: It loads the master dataset from the specified .npy file using the `open_npy_file` function.

2. **Extracting Features and Target Variables**: The script separates the input features (X) and target variables (y) from the loaded dataset.

3. **Integer-to-Binary Conversion**: It includes a function `integer_to_binary_array` to convert integer target variables to binary arrays, which is essential for multi-output classification.

4. **Mapping Labels to Indices**: The script creates mappings between original labels and class indices for classification purposes.

5. **Replacing Labels with Class Indices**: It modifies the target variables to utilize class indices instead of original labels, optimizing the dataset for classification tasks.

6. **Counting Label Occurrences**: The script calculates the occurrences of each label in the dataset, providing insights into class distribution.

7. **Filtering Dataset**: Instances where the agent solely moves forward are filtered out to prevent misclassification, particularly in multi-output multiclass training scenarios. This filtering is crucial to ensure balanced representation of various actions, such as jumping, at critical states.

8. **Saving Filtered Dataset**: The filtered dataset, free from instances solely characterized by forward movement, is saved as a new .npy file for further analysis or training.

## Usage

To use the script:
- Ensure that the path to the master dataset `.npy` file is correctly specified.
- Run the script to preprocess the data, filter out specific instances, and save the filtered dataset.

## Output

By removing instances solely characterized by forward movement, the script aims to enhance the dataset's suitability for multi-output multiclass classification tasks, ensuring accurate model learning.
