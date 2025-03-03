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

import os

def main():
    # Get the current directory
    current_directory = os.getcwd()
    # Define the directory containing .npy files
    directory = "/Users/sankalpssss/Documents/marioenv/mario/mario2/Data final/Image/All_Img_session"
    # List all .npy files in the directory
    npy_files = [file for file in os.listdir(directory) if file.endswith('.npy')]

    #All sessions combined: 
    session_list_state = []
    session_list_actions = []

    # Iterate over each .npy file
    for file_name in npy_files:
        # Construct the full file path
        file_path = os.path.join(directory, file_name)
        # Load data from the .npy file
        loaded_data = open_npy_file(file_path)

        for i in range(loaded_data[0].shape[0]):
            #Append the states in session_list_state and the actions resoectively
            session_list_state.append(loaded_data[0][i])
            session_list_actions.append(loaded_data[1][i])

    final_state_array = np.empty((len(session_list_state),),dtype=object)
    for i in range(len(session_list_state)):
        final_state_array[i] = session_list_state[i]

    final_action_array = np.empty((len(session_list_actions),),dtype=object)
    for i in range(len(session_list_actions)):
        final_action_array[i] = session_list_actions[i]

    dataset = np.array((final_state_array,final_action_array))
    np.save('/Users/sankalpssss/Documents/marioenv/mario/mario2/Data final/Image/Master_image',dataset)

    #view the dataset shapes
    print("----------------------------------------")
    print("1.) Combined Dataset size",dataset.shape)


    print("----------------------------------------")
    print("2.) Training set X (RAM states):")
    print(dataset[0].shape) #Collection of flattened arrays
    #print(dataset[0][0].shape) #state output flattened
    #print(dataset[0][0][0].shape) #the numebr itself from 13x13

    print("----------------------------------------")
    print("3.) Training set Y(Action):")
    print(dataset[1].shape) #Collection of flattened arrays
    #print(dataset[1][0].shape) #action flattened
    #print(dataset[1][0][0].shape) #the numebr itself from 13x13

if __name__ == "__main__":
    main()