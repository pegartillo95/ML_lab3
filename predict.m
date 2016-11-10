function p = predict(Theta1, Theta2, X)
m = size(X, 1);
num_labels = size(Theta2, 1);
p = zeros(size(X, 1), 1);

% Add ones to the X data matrix
X = [ones(m, 1) X];

% Coverts to matrix of 5000 examples x 26 thetas
z1=X*Theta1';
% Sigmoid function converts to p between 0 to 1
h1=sigmoid(z1);

% Add ones to the h1 data matrix
h1=[ones(m, 1) h1];
% Converts to matrix of 5000 exampls x num_labels 
z2=h1*Theta2';
% Sigmoid function converts to p between 0 to 1
h2=sigmoid(z2);

% pval returns the highest value in each row, while p returns the position in each row
[pval, p]=max(h2,[],2);  

end