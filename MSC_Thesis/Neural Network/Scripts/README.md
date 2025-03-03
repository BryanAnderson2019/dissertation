# Neural Network Training Summary:

Script: [ANN.py](https://github.com/sankalp-s/MSC_Thesis/blob/main/Neural%20Network/Scripts/ANN.py)

#### Dataset:
- Loaded a dataset containing features and labels for training a neural network.
- The dataset consists of flattened 13x13 extracted RAM arrays as features and single integer values representing actions as labels.

#### Model Architecture:
- Implemented a Sequential neural network model using TensorFlow and Keras.
- Utilized Dense layers with ReLU activation for hidden layers and a Softmax activation for the output layer.
- The model architecture includes three hidden layers with 100, 20, and 40 neurons respectively.

#### Training:
- Compiled the model using the Adam optimizer and Sparse Categorical Crossentropy loss function.
- Trained the model for 150 epochs with a batch size of 10.
- Implemented early stopping with a patience of 10 to prevent overfitting.

#### Saved Model:
- Saved the trained model for future use.
