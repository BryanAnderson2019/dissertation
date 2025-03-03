import numpy as np
import os

def open_npy_file(file_path):
    try:
        data = np.load(file_path, allow_pickle=True)
        return data
    except FileNotFoundError:
        print("File not found.")
        return None
    except Exception as e:
        print("An error occurred:", e)
        return None
    
def binary_array_to_integer(binary_array):
    """
    Convert a binary array to an integer.
    Parameters:
        binary_array (list): The binary array to be converted.
    Returns:
        int: The integer value obtained from the binary array.
    """
    binary_string = ''.join(map(str, binary_array))
    decimal_integer = int(binary_string, 2)
    return decimal_integer

import os

def main():
    # Get the current directory
    current_directory = os.getcwd()
    # Define the directory containing .npy files
    directory = "/Users/sankalpssss/Documents/marioenv/mario/mario2/Data final/All"
    # List all .npy files in the directory
    npy_files = [file for file in os.listdir(directory) if file.endswith('.npy')]

    #All sessions combined: 
    session_list = []

    # Initialize a variable to store the sum of the first indices
    sum_first_indices = 0

    # Iterate over each .npy file
    for file_name in npy_files:
        # Construct the full file path
        file_path = os.path.join(directory, file_name)
        # Load data from the .npy file
        loaded_data = open_npy_file(file_path)

        # Get the shape of the loaded data
        data_shape = loaded_data[0].shape
        # Sum up the first index of the shape
        sum_first_indices += data_shape[0]

        # Convert binary arrays to integers
        Integer_array = np.array([binary_array_to_integer(binary_array) for binary_array in loaded_data[1]])

        # Append state space and action space arrays
        appended_arrays = []
        for i in range(len(Integer_array)):
            appended_array = np.append(loaded_data[0][i], Integer_array[i])
            appended_arrays.append(appended_array)

        appended_arrays = np.array(appended_arrays)

        #Collecting all the converted sessions to a list
        session_list.append(appended_arrays)

        print("Resulting array shape for file", file_name, ":", appended_arrays.shape)

        #Create the final data set by vertically stacking the sesion from the list
        L1_combined_session_data = np.vstack(session_list)

    print(sum_first_indices)

    print("Final Level1 dataset shape:", L1_combined_session_data.shape)
    np.save('/Users/sankalpssss/Documents/marioenv/mario/mario2/Data final/Combined_dataset/Integer/Master2_integer', L1_combined_session_data)
        

if __name__ == "__main__":
    main()