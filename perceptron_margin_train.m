% This function implements the perceptron with margin algorithm on the
% training set.
%
% Input:
% x: k-by-n matrix,
% y: k-by-1 vector, each element can be 1 or -1
% r: learning rate.
%
% Output:
% w: 1-by-n vector of the weight vector.
% theta: the theta weight parameter corresponding to w_0.

function [w,theta] = perceptron_margin_train(x,y,r)
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
            if(y(i)*(product_i + theta) <= 1)
                %Update the weight vector and theta.
                for j = 1:n
                    w(j) = w(j) + r*y(i)*x(i,j);
                end
                theta = theta + r*y(i);
            end
        end
    end
        