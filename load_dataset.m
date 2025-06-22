% 312CA_Nitu-Eriko-Laurentiu
function [X, y] = load_dataset(path)
	% path -> a relative path to the .mat file that must be loaded
	% X, y -> the training examples (X) and their corresponding labels (y)
	
	% loading the data from the file
	load(path);
	
end
