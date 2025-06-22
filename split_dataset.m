% 312CA_Nitu-Eriko-Laurentiu
function [X_train, y_train, X_test, y_test] = split_dataset(X, y, percent)
	% X -> the loaded dataset with all training examples
	% y -> the corresponding labels
	% percent -> fraction of training examples to be put in training dataset
	% X_[train|test] -> the datasets for training and test respectively
	% y_[train|test] -> the corresponding labels
	% Example: [X, y] has 1000 training examples with labels and percent = 0.85
	%           -> X_train will have 850 examples
	%           -> X_test will have the other 150 examples

	% getting the number of examples
	[nr_examples, ~] = size(X);

	% shuffling the rows in the examples matrix and the labels vector
	perm = randperm(nr_examples);
	X = X(perm, :);
	y = y(perm);

	% splitting the examples and the corresponding labels according to the
	% given percent
	nr_rows_train = floor(percent * nr_examples);
	nr_rows_test = nr_examples - nr_rows_train;
	X_train = X(1:nr_rows_train, :);
    y_train = y(1:nr_rows_train);

    X_test = X(nr_rows_train + 1 : end, :);
    y_test = y(nr_rows_train + 1 : end);	

end
