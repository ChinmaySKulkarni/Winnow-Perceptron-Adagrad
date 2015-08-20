% This function implements the perceptron algorithm on the training set.
%
% Input:
% x: k-by-n matrix,
% y: k-by-1 vector, each element can be 1 or -1
%
% Output:
% w: 1-by-n vector of the weight vector.
% theta: the theta weight parameter corresponding to w_0.

function [w,theta] = perceptron_train(x,y)
    [k,n] = size(x);
    %The weight vector is 'w'. 
    %Initializing the weight vector and theta to zeros.
    w = zeros(1,n);   
    theta = 0;
    
    for iterate = 1:20
        for i = 1:k
            %Select the ith training example.
            training_example = x(i,:);
            product_i = dot(w,training_example);
            if(y(i)*(product_i + theta) <= 0)
                %Update the weight vector and theta.
                for j = 1:n
                    w(j) = w(j) + 1*y(i)*x(i,j);
                end
                theta = theta + 1*y(i);
            end
        end
    end