function [C, sigma] = dataset3Params(X, y, Xval, yval)
%EX6PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = EX6PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

C_can = [0.01 0.03 0.1 0.3 1 3 10 30]';
sigma_can = [0.01 0.03 0.1 0.3 1 3 10 30]';
error_val = zeros(length(C_can), length(sigma_can));

for i = 1:length(C_can)
    for j = 1:length(sigma_can)
        model = svmTrain(X, y, C_can(i), @(x1, x2) gaussianKernel(x1, x2, sigma_can(j)));
        predictions = svmPredict(model, Xval);
        error_val(i, j) = mean(double(predictions ~= yval));
    end
end

[~, I] = min(error_val(:));
[I_row, I_col] = ind2sub(size(error_val), I);
C = C_can(I_row);
sigma = sigma_can(I_col);

% =========================================================================

end
