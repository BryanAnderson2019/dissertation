import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Load the entire dataset
data = np.load("/Users/sankalpssss/Documents/marioenv/mario/mario2/Data final/Image/Master_image.npy", allow_pickle=True)

X = data[0]
y = data[1]

X = np.array([np.array(val) for val in X])  # Fixes issues with numpy loading
X = np.array([np.expand_dims(val,0) for val in X])  # Add virtual batch to start

print(X.shape)

y = np.array([np.array(val) for val in y])  # Fixes issues with numpy loading
y = np.array([val.reshape(1,12) for val in y])  # Reshape to fit model

print(y.shape)

X = X / 255  # Normalize image

# Convert TensorFlow dataset to numpy arrays
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Split the training data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42)

"""

# Print shapes to verify the split
print("X_train shape:", X_train.shape)
print("X_val shape:", X_val.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_val shape:", y_val.shape)
print("y_test shape:", y_test.shape)

train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))#Make into dataset
test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test))#Make into dataset
validation_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val))#Make into dataset


callback = tf.keras.callbacks.EarlyStopping(patience=10)#Stop if validation accuracy goes down, prevents overfitting


#Model obtained from https://www.tensorflow.org/tutorials/images/cnn
model = Sequential()
model.add(Conv2D(16, (3, 3), activation='relu', input_shape=(84,84,1)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(12, activation='sigmoid'))
model.summary()

model.compile(loss='BinaryCrossentropy', optimizer="adam", metrics=['accuracy'])
history=model.fit(train_dataset, epochs=150, batch_size=8,verbose=2,validation_data=(validation_dataset),callbacks = [callback])

model.summary()
tf.keras.models.save_model(model,'/Users/sankalpssss/Documents/marioenv/mario/mario2/Models/Final Models/ANN_Image')

#summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# Evaluate model on test data
loss, accuracy = model.evaluate(X_test, y_test)
print('Test Accuracy:', accuracy)
print('Test Loss:', loss)

"""