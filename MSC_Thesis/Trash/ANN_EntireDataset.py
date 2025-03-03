# Import necessary libraries
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
import matplotlib.pyplot as plt

# Load data
combined_data = np.load("/Users/sankalpssss/Documents/marioenv/mario/mario2/Data final/Combined_dataset/Integer/Master_integer.npy", allow_pickle=True)
# Separate features and labels
X = combined_data[:, :-1]
y = combined_data[:, -1] #Last column is the target variable

#Shape of the dataset
print(combined_data.shape)

#Shape of the training dataset X
print(X.shape) #It is flattened 13x13 extracted RAM arrays

#Shape of the target variable dataset
print(y.shape) #It contains single integer values obtained from converting the binary array(of the action)

# Extract unique integer labels
original_labels = np.unique(y)

# Create a mapping between integer labels and class indices which will be later used in classification as classes.
label_to_index = {label: index for index, label in enumerate(original_labels)}

# Create a mapping from class indices to original labels which will be later used to track the labels from the class index.
index_to_label = {index: label for label, index in label_to_index.items()}

# Replace labels with corresponding class indices
# Modifying the target variables for classification
y_categorical = np.array([label_to_index[label] for label in y])

print(original_labels)
print(label_to_index)

# Determine the number of classes
num_classes = len(original_labels)

# Define model
#Softmax activation is commonly used in multi-class classification problems as it outputs a probability distribution over the classes.

model = Sequential([
    Flatten(input_shape=(X.shape[1],)),  # Input shape depends on the number of features
    Dense(100, activation='relu'),
    Dense(20, activation='relu'),
    Dense(40, activation='relu'),
    Dense(num_classes, activation='softmax')  # Output layer with softmax activation
])

model.summary()

#Stop if validation accuracy goes down, prevents overfitting
callback = tf.keras.callbacks.EarlyStopping(patience = 10)

# Compile model
# Sparse categorical cross-entropy loss function is commonly used for classification problems where the labels are integers.
model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])

# Train model
history = model.fit(X, y_categorical, epochs=150, batch_size=10, verbose=2)

tf.keras.models.save_model(model,'/Users/sankalpssss/Documents/marioenv/mario/mario2/Models/Final Models/ANN_EntireDataset')

# Plot accuracy
plt.plot(history.history['accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.show()

# Plot loss
plt.plot(history.history['loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.show()

"""# Use the trained model to make a prediction about a single state."""

Predict = np.load("/Users/sankalpssss/Documents/marioenv/mario/mario2/Data final/Level1/AnshikalL1.npy", allow_pickle=True) #Sample data (same dimensions as the main dataset)
X_predict = Predict[:, :-1]
y_predict = Predict[:, -1]

single_state  = (np.expand_dims(X_predict[50],0)) # Add it to a batch where it's the only member.

print(X_predict[0].shape) #Intial shape of the state in the dataset
print(single_state.shape) #Desired state for the predictions using the model
single_predict = model.predict(single_state) #Predicting the output from the single state

index_single = np.argmax(single_predict) #Finding the label that has the highest confidence value
final_action_integer_single = index_to_label[index_single] #Converting the obtained index to the label
print(final_action_integer_single) #Print the action which is encoded to an integer

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
    return binary_array

binary_data = integer_to_binary_array(final_action_integer_single) #Converting the action integer to the binary array which acts as the actual input to the game
print(binary_data)
