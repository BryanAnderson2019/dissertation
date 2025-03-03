# Script

[Model_Play_ANN.py](https://github.com/sankalp-s/MSC_Thesis/blob/main/Neural%20Network/Replay_model/Model_Play_ANN.py)

This script utilizes a trained artificial neural network (ANN) model to predict actions in the Super Mario World game environment. Here's a breakdown of its functionality:

- **Model Loading:** The script loads a pre-trained ANN model saved in the specified file path using TensorFlow.
  
- **Environment Setup:** It sets up the Super Mario World game environment using the Retro library, specifically focusing on the YoshiIsland2 state.

- **Action Prediction:** The `predict_action` function takes the current game state as input and predicts the next action to take based on the learned patterns from the trained model. It converts the predicted action from integer format to a binary array format.

- **Action Execution:** The predicted action is executed in the game environment, and the game state is rendered to observe the gameplay. Additionally, a blank action is sent to prevent the agent from holding an action continuously.

This script serves as a demonstration of using machine learning models to control game agents, showcasing the application of ANN in decision-making processes within game environments.
