% 312CA_Nitu-Eriko-Laurentiu
function [J, grad] = cost_function(params, X, y, lambda, ...
				   input_layer_size, hidden_layer_size, ...
				   output_layer_size)
	% params -> vector containing the weights from the two matrices
	%           Theta1 and Theta2 in an unrolled form (as a column vector)
	% X -> the feature matrix containing the training examples
	% y -> a vector containing the labels (from 1 to 10) for each
	%      training example
	% lambda -> the regularization constant/parameter
	% [input|hidden|output]_layer_size -> the sizes of the three layers
	% J -> the cost function for the current parameters
	% grad -> a column vector with the same length as params
	% These will be used for optimization using fmincg

	[m, ~] = size(X);

	% reshaping the weights vector in two matrices, one for the input layer
	% and the next one for the hidden layer
	Theta1 = reshape(params(1:(hidden_layer_size * (input_layer_size + 1))), ...
	                 hidden_layer_size, input_layer_size + 1);
	Theta2 = reshape(params((hidden_layer_size * (input_layer_size + 1)) + 1:end), ...
	                 output_layer_size, hidden_layer_size + 1);

	% performing the forward propagation
	a1 = [ones(m, 1), X];
	z2 = a1 * Theta1';

	a2 = sigmoid(z2);
	a2 = [ones(m, 1), a2];

	z3 = a2 * Theta2';
	a3 = sigmoid(z3);
	h_theta = a3; % the prediction matrix
 

	% transforming the labels vector in a matrix
	% each row has 1 in the collumn of the correct class and 0 in rest
	y_matrix = eye(output_layer_size)(y,:);

	% calculating the regularised cost function
	cost = -y_matrix .* log(h_theta) - (1 - y_matrix) .* log(1 - h_theta);
	regularization = (lambda / (2 * m)) * (sum(Theta1(:, 2:end)(:) .^2) + ...
	                 sum(Theta2(:, 2:end)(:) .^ 2));
	J = (1 / m) * sum(cost(:)) + regularization;
	
	% determining the gradients of the weight matrices with backpropagation
	Delta1 = zeros(size(Theta1));
	Delta2 = zeros(size(Theta2));

	% error in the ouput layer
	err_3 = a3 - y_matrix;
	% multiplying the error with the activations from the previous layer
	Delta2 = err_3' * a2;

	% error in the hidden layer:
	% backpropagating the error through the weights (excluding the bias) and
	% multiplying term by term with the derivative of the sigmoid function in
	% order to see how much did each neuron from the hidden layer contributed
	% to the error and how sensitive is the cost function to changes in z2 
	err_2 = (err_3 * Theta2(:, 2:end)) .* (sigmoid(z2) .* (1 - sigmoid(z2)));
	% multiplying the error with the activations from the previous layer
	Delta1 = err_2' * a1;

	% calculating the gradient and adding the regularisation
	Theta1_gradient = (1 / m) * Delta1;
	Theta2_gradient = (1 / m) * Delta2;
	Theta1_gradient(:, 2:end) += (lambda / m) * Theta1(:, 2:end);
	Theta2_gradient(:, 2:end) += (lambda / m) * Theta2(:, 2:end);

	% saving the gradients in a vector
	grad = [Theta1_gradient(:); Theta2_gradient(:)];

end
