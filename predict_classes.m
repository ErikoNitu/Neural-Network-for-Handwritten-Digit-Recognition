% 312CA_Nitu-Eriko-Laurentiu
function [classes] = predict_classes(X, weights, ...
				  input_layer_size, hidden_layer_size, ...
				  output_layer_size)
	% X -> the test examples for which the classes must be predicted
	% weights -> the trained weights (after optimization)
	% [input|hidden|output]_layer_size -> the sizes of the three layers
	% classes -> a vector with labels from 1 to 10 corresponding to
	%            the test examples given as parameter
	% înainte de predict

	m = size(X, 1);

	% reshaping the weights vector in two matrices, one for the input layer
	% and the next one for the hidden layer
	Theta1 = reshape(weights(1:(hidden_layer_size * (input_layer_size + 1))), hidden_layer_size, input_layer_size + 1);
	Theta2 = reshape(weights((hidden_layer_size * (input_layer_size + 1)) + 1:end), output_layer_size, hidden_layer_size + 1);
	
	% performing the forward propagation
	a1 = [ones(m, 1), X];
	z2 = a1 * Theta1';

	a2 = sigmoid(z2);
	a2 = [ones(m, 1), a2];

	z3 = a2 * Theta2';
	a3 = sigmoid(z3);

	[m, n] = size(a3);
	classes = zeros(m, 1);

	% extracting the maximum "probability" from each row to determine
	% the class
	for i = 1:m
		max_val = -inf;
		max_idx = 0;

		for j = 1:n
			if a3(i, j) > max_val
				max_val = a3(i, j);
				max_idx = j;
			end
		end
		classes(i) = max_idx;
	end
	disp(classes);
end
