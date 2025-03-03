
import numpy as np
import matplotlib.pyplot as plt

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

def integer_to_binary_array(integer_value, array_length=12):
    """
    Convert an integer to a binary array of specified length.
    Parameters:
        integer_value (int): The integer value to be converted.
        array_length (int): The desired length of the binary array.
    Returns:
        list: The binary array representing the integer value.
    """
    binary_string = format(integer_value, 'b')  # Convert integer to binary string
    binary_array = [int(bit) for bit in binary_string.zfill(array_length)]  # Pad with leading zeros if needed
    return np.array(binary_array)

# Example usage
file_path = '/Users/sankalpssss/Documents/marioenv/mario/mario2/Data final/Combined_dataset/Integer/Master_integer.npy'
loaded_data = open_npy_file(file_path)

if loaded_data is not None:
    print("----------------------------------------")
    print(loaded_data[:1])  # Print first 5 elements
    print("1.) Combined Dataset size", loaded_data.shape)
    print("----------------------------------------")

X = loaded_data[:, :-1]
y = loaded_data[:, -1] #Last column is the target variable

# Extract unique integer labels
original_labels = np.unique(y)

# Create a mapping between integer labels and class indices which will be later used in classification as classes.
label_to_index = {label: index for index, label in enumerate(original_labels)}

# Create a mapping from class indices to original labels which will be later used to track the labels from the class index.
index_to_label = {index: label for label, index in label_to_index.items()}

# Replace labels with corresponding class indices
# Modifying the target variables for classification
y_categorical = np.array([label_to_index[label] for label in y])

# Count occurrences of each label
label_counts = np.unique(y, return_counts=True)

print("Label Counts:")
for label, count in zip(label_counts[0], label_counts[1]):
    print(f"Label {label}: {count} times")

print('Label Values:')
for label in original_labels:
    print('label', label, 'binary array', integer_to_binary_array(label))

#create a new dataset by removing all the instances where the value is zero and 16
filtered_indices = np.where((y != 16))[0]
filtered_dataset = loaded_data[filtered_indices]

np.save('/Users/sankalpssss/Documents/marioenv/mario/mario2/Data final/Combined_dataset/Integer/FilteredData.npy', filtered_dataset)

print('Filtered Dataset size:', filtered_dataset.shape)
