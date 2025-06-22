% 312CA_Nitu-Eriko-Laurentiu
function [matrix] = initialize_weights(L_prev, L_next)
	% L_prev -> the number of units in the previous layer
	% L_next -> the number of units in the next layer
	% matrix -> the matrix with random values
	
	% calculating epsilon using the number of units from the input and
	% hidden layers
	epsilon = sqrt(6) / sqrt(L_next + L_prev);

	% initialising the matrix with random walues from [-epsilon, epsilon]
	matrix = 2 * epsilon * rand(L_next, L_prev + 1) - epsilon;
	
end
