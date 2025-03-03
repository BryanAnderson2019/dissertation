# Classifiers Overview

This section provides an overview of the classifiers utilized in the project, along with their respective Python scripts.

## Decision Tree Classifier

**Script:** [decisiontree_classifier.py](https://github.com/sankalp-s/MSC_Thesis/blob/main/Classifiers/DecisionTree/decisiontree_classifier.py)

The Decision Tree Classifier is a non-parametric supervised learning method used for classification tasks. It works by recursively partitioning the input space into regions based on feature values, with the aim of minimizing impurity or maximizing information gain at each split. The script implements the Decision Tree Classifier algorithm, providing functionality for training and making predictions based on input features.

## Gradient Boosting Classifier

**Script:** [gradientboosting.py](https://github.com/sankalp-s/MSC_Thesis/blob/main/Classifiers/GradienBoosting/gradientboosting.py)

The Gradient Boosting Classifier is an ensemble learning method that combines the predictions of multiple weak learners, typically decision trees, to improve overall predictive accuracy. It builds a series of decision trees sequentially, with each tree correcting the errors of the previous ones. The script implements the Gradient Boosting Classifier algorithm, offering functionalities for training and prediction.

## K-Nearest Neighbors (KNN) Classifier

**Script:** [kneighborsclassifier.py](https://github.com/sankalp-s/MSC_Thesis/blob/main/Classifiers/KNeighborsClassifier/kneighborsclassifier.py)

The K-Nearest Neighbors (KNN) Classifier is a simple and intuitive machine learning algorithm used for classification tasks. It classifies a data point based on the majority class among its k nearest neighbors in the feature space. The script provides an implementation of the KNN algorithm, allowing for training and prediction based on the nearest neighbors of input samples.

## Linear Support Vector Classifier (LinearSVC)

**Script:** [linearsvc.py](https://github.com/sankalp-s/MSC_Thesis/blob/main/Classifiers/LinearSVC/linearsvc.py)

The Linear Support Vector Classifier (LinearSVC) is a linear classifier that separates classes by defining a hyperplane in the feature space. It aims to maximize the margin between different classes while minimizing classification errors. The script implements the LinearSVC algorithm, offering functionalities for training and prediction.

## Random Forest Classifier

**Script:** [randomforestclassifier.py](https://github.com/sankalp-s/MSC_Thesis/blob/main/Classifiers/Random_Forest/randomforestclassifier.py)

The Random Forest Classifier is an ensemble learning method that builds a collection of decision trees during training and combines their predictions through voting or averaging to improve accuracy and robustness. It is particularly effective for handling high-dimensional data and complex classification tasks. The script implements the Random Forest Classifier algorithm, providing functionalities for training and prediction.

# Folders

## Trained Models

[Trained_Models](https://github.com/sankalp-s/MSC_Thesis/tree/main/Classifiers/Trained_Models)

This folder contains the trained models generated using scripts for various classifiers. The models are trained to classify actions in the Super Mario World game environment based on the extracted features.
You can load these trained models into your Python environment using TensorFlow or scikit-learn library to perform classification tasks on new data.

## Results

[Results](https://github.com/sankalp-s/MSC_Thesis/tree/main/Classifiers/Results)

The `results` folder contains comparison graphs illustrating the accuracies of various models used in the project. These graphs provide insights into the performance of different models and their suitability for the task at hand.

The comparison graphs serve as a visual aid for evaluating and selecting the most effective model for the given task. By analyzing the accuracy trends and performance differences among models, stakeholders can make informed decisions regarding model selection and optimization strategies.


# Scripts

[Model_play.py](https://github.com/sankalp-s/MSC_Thesis/blob/main/Classifiers/Model_play.py)

This script demonstrates how to use a trained machine learning model to control an agent playing the Super Mario World game in a Retro environment.

## Overview

The script performs the following tasks:

1. **Model Loading**: Loads a trained machine learning model saved using the joblib library. In this example, a RandomForest model is loaded from a file.

2. **Environment Setup**: Sets up the Retro environment for playing the Super Mario World game. The desired game and state (level) are specified, and the observation type is set to image.

3. **Action Prediction**: Defines a function to predict the action to take based on the current game state. The state is obtained from the RAM of the game environment and is processed to prepare it for input to the model. The model then predicts the action to take.

4. **Action Execution**: Converts the predicted action into the appropriate format (binary array) and applies it to the game environment using the `env.step()` function. The environment is rendered to display the game state.

5. **Loop**: The process repeats in a loop, continuously predicting and executing actions until the game is completed or terminated.

This script provides a framework for automated gameplay using a trained machine learning model. It allows the agent to navigate and interact with the game environment in real-time, showcasing the potential of machine learning techniques in gaming applications.



