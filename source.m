% Load training data
data = load('training_2.txt'); 
X = data(:, 1:2);
y = data(:, 3);
m = length(y);

% Add intercept term to X
X = [ones(m, 1) X];

% Calculate the parameters from the normal equation
theta = inv(X'*X)*(X'*y);

% Display normal equation's result
fprintf('Theta computed from the normal equations: \n');
fprintf(' %f \n', theta);
temp = 1/(2*m)*sum((X*theta-y).^2); %compute cost
fprintf('cost = %f\n',temp); %cost here
fprintf('\n');

% Load test data
test_data = load('test_2.txt');

% Add intercept term
test_data = [ones(length(test_data), 1) test_data];

% Write to sample_submission
test_prediction = test_data*theta;
csvwrite('sample_submission.csv',test_prediction);
