import numpy as np

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


# Example usage
file_path = '/home/bryan/dissertation/Project/Player_Inputs/Dataset/best_runs/best_run_LONG_PD.npy'  # Replace 'data.npy' with the path to your .npy file
loaded_data = open_npy_file(file_path)

if loaded_data is not None:
    print("Loaded data state:")
    print(loaded_data.shape)
    print(loaded_data[0].shape) #Collection of flattened arrays
    print(loaded_data[0][0].shape) #state output flattened
    print(loaded_data[0][0][0].shape) #single element

    print("Loaded data action:")
    print(loaded_data.shape)
    print(loaded_data[1].shape) #Collection of flattened arrays
    print(loaded_data[1][0].shape) #action output flattened
    print(loaded_data[1][0][0].shape) #single element
