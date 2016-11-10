function p = predictOneVsAll(all_theta, X)

m = size(X, 1);
num_labels = size(all_theta, 1);
 
p = zeros(size(X, 1), 1);

% Add ones to the X data matrix
X = [ones(m, 1) X];

% Coverts to matrix of 5000 examples vs. num_lables (each sample has num_labels of corresponding prob
z=X*all_theta';
% Sigmoid function converts to p between 0 to 1
h=sigmoid(z);

% pval returns the highest value in each row, while p returns the position in each row
[pval, p]=max(h,[],2);  

end
