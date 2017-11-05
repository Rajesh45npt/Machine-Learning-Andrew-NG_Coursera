function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);
n= size(theta,1);
h = zeros(m,1);
temp = zeros(n,1);
temp_theta=theta;
% figure;
% hold on;

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %
    
    h = X*theta;
    del = h-y;
    
    for loopnum = 1:n
        
        temp(loopnum,1) = sum(del .* X(:, loopnum));
    end
    
    temp = temp / m;
    
    theta = theta - alpha* temp;
    
     
    





    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);
    %plot(iter,J_history(iter),'r+', 'MarkerSize',1);

end

end
