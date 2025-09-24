
import numpy as np



# this function initializes the weights and biases for our 2 layer neural network
def initialise_weights(input_size, hidden_size, output_size):
  np.random.seed(16)  
  parameters = {    
    # initialize weights with small random values
    "W1": np.random.randn(input_size, hidden_size) * 0.01,
    "b1": np.zeros((hidden_size, 1)),
    "W2": np.random.randn(hidden_size, output_size) * 0.01,
    "b2": np.zeros((output_size, 1))
  }
  return parameters

# for output layer
def sigmoid(x):
  x = np.clip(x, -500, 500) # this keeps the numbers from getting too big and causing warnings
  return 1 / (1 + np.exp(-x))

# for hidden layer
def relu(x):
  return 1/(1 + np.exp(-x))

# the purose of this function is to return 1 if x>0 and 0 if x<=0
def relu_derivitive(x):
  return (x>0).astype(int)