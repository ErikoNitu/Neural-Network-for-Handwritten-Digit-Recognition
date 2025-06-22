This programme involves training a neural network to recognize handwritten digits
from 20x20 pixel images. The network architecture includes an input layer,
a hidden layer and two weight matrices that are optimized during training.
After loading the dataset and splitting it into training and test examples,
I initialized the weight matrices with random values in the interval 
[-epsilon, epsilon], where epsilon was calculated based on the sizes of the
input and hidden layers.
I then performed forward propagation to obtain the predicted class
probabilities, calculated the cost function with regularization and applied
backpropagation to calculate the gradients of the weight matrices. These
gradients were used to optimize the weights using the fmincg algorithm.
Finally, I performed forward propagation again on the test set and selected
the maximum value of each row of the output layer representing the predicted
class for each example.

- function [X, y] = load_dataset(path)
  Loads a mat file from the given path. Returns a matrix X where each row is
  a training example and y with the corresponding labels for the examples.

- function [X_train, y_train, X_test, y_test] = split_dataset(X, y, percent)
  Randomly shuffles the examples and splits it into training and testing sets
  based on the given percent.
  Next we are going to work with X_train and y_train.

- function [matrix] = initialize_weights(L_prev, L_next)
  Initializes the weights between two neural network layers with small random
  values from the interval (−epsilon, epsilon) so to break symetry.
  We do not initialize weights with zero because it would cause all neurons
  in the same layer to learn the same features during training.

- function [J, grad] = cost_function(params, X, y, lambda, input_layer_size,
                                     hidden_layer_size, output_layer_size)
  In this function I reshaped the weight matrices with values from the params
  vector, performed the forwardpropagation through the layers in order to
  obtain the predicted class probabilities and calculated the cost function
  based on it. After that, I applied backpropagation to calculate the gradients
  of the weight matrices, which are later used to optimize the network
  parameters.

- function [classes] = predict_classes(X, weights, input_layer_size,
                                       hidden_layer_size, output_layer_size)
  Performed forward propagation to calculate the predicted class labels for a
  given input set and trained weights then I extracted the maximum value from
  each row representing the probability of the example to represent a certain
  class.

