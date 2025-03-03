# Removing Duplicate Rows from the Dataset
[Preprocessing_duplicates.py](https://github.com/sankalp-s/MSC_Thesis/blob/main/Preprocessing/Preprocessing_duplicates.py)

The provided script removes duplicate rows from a dataset. Here's a breakdown of the code:

## Functions
- `open_npy_file(file_path)`: Loads data from a .npy file, handling potential errors such as file not found or other exceptions.
- `remove_duplicates(data)`: Identifies and removes duplicate rows from the input dataset.

## Example Usage
- The script loads a dataset from a .npy file located at a specific file path and prints its original shape.
- Duplicate rows are removed using the `remove_duplicates` function.
- The script prints the shape of the dataset after removing duplicates and the total number of duplicate rows found.


# Removing insignificant instances
[Preprocessing_remove_instance.py](https://github.com/sankalp-s/MSC_Thesis/blob/main/Preprocessing/Preprocessing_remove_instance.py)

The provided script loads a dataset from a .npy file and prepares it for further processing.

## Functions
- `open_npy_file(file_path)`: Loads data from a .npy file, handling potential errors such as file not found or other exceptions.
- `integer_to_binary_array(integer_value, array_length=12)`: Converts an integer value to a binary array of specified length.

## Data Processing
- The loaded dataset is separated into feature matrix `X` and target vector `y`.
- Unique integer labels present in the target vector `y` are extracted.
- A mapping is created between the original integer labels and class indices for classification.

## Dataset Filtering

| Class/Label | Action Taken        | Binary Array                        |
|-------------|---------------------|-------------------------------------|
| 0           | No action           | [0 0 0 0 0 0 0 0 0 0 0 0]           |
| 8           | Spin Jump           | [0 0 0 0 0 0 0 0 1 0 0 0]           |
| 16          | Move Right          | [0 0 0 0 0 0 0 1 0 0 0 0]           |
| 24          | Right + Spin Jump   | [0 0 0 0 0 0 0 1 1 0 0 0]           |
| 32          | Left                | [0 0 0 0 0 0 1 0 0 0 0 0]           |
| 40          | Left + Spin Jump    | [0 0 0 0 0 0 1 0 1 0 0 0]           |
| 48          | Left + Right        | [0 0 0 0 0 0 1 1 0 0 0 0]           |
| 56          | Left + Right + Spin Jump | [0 0 0 0 0 0 1 1 1 0 0 0]     |
| 64          | Down                | [0 0 0 0 0 1 0 0 0 0 0 0]           |
| 80          | Down + Right        | [0 0 0 0 0 1 0 1 0 0 0 0]           |
| 144         | Up + Right          | [0 0 0 0 1 0 0 1 0 0 0 0]           |
| 2048        | Jump                | [1 0 0 0 0 0 0 0 0 0 0 0]           |
| 2064        | Jump + Right        | [1 0 0 0 0 0 0 1 0 0 0 0]           |
| 2072        | Jump + Spin + Right| [1 0 0 0 0 0 0 1 1 0 0 0]           |
| 2080        | Jump + Left         | [1 0 0 0 0 0 1 0 0 0 0 0]           |
| 2096        | Jump + Left + Right| [1 0 0 0 0 0 1 1 0 0 0 0]           |


- Instances in the dataset where the target value is either 0, 16, 32, 40, 48, 56, 80, 2080, or 2096 are filtered out and removed.
- The filtered dataset is saved to a new .npy file.
- The script prints the size of the filtered dataset.

This script serves as a template for loading, processing, and filtering datasets stored in .npy files, facilitating data preprocessing tasks in machine learning projects.

