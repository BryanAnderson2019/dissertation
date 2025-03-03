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

def remove_duplicates(data):

    unique_rows, counts = np.unique(data, axis=0, return_counts=True)
    duplicate_indices = np.where(counts > 1)[0]
    total_duplicates = np.sum(counts[duplicate_indices])
    print("Total duplicate rows found:", total_duplicates)

    # Get the indices of unique rows
    _, unique_indices = np.unique(data, axis=0, return_index=True)

    # Extract unique rows
    unique_data = data[unique_indices]

    return unique_data

# Example usage
file_path = '/Users/sankalpssss/Documents/marioenv/mario/mario2/Data final/Combined_dataset/Integer/Master_integer.npy'
loaded_data = open_npy_file(file_path)
print("Original Dataset Shape", loaded_data.shape)

if loaded_data is not None:
    # Remove duplicate rows
    unique_data = remove_duplicates(loaded_data)
    print("Dataset after removing duplicates", unique_data.shape)

    """
    # Save the new dataset
    new_file_path = '/Users/sankalpssss/Documents/marioenv/mario/mario2/Data final/Combined_dataset/Integer/Dataset_Wdupli.npy'
    np.save(new_file_path, unique_data)
    print("New dataset saved successfully as", new_file_path)
    """
