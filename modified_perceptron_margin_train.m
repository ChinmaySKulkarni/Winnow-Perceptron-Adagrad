% This function implements the perceptron with margin algorithm on the
% training set. This algorithm is modified to handle unbalanced datasets.
% The modification is in the value of the margin.
% The margin is taken in the inverse proportion to the proportion of the
% data in the unbalanced data set.
%
% Input:
% x: k-by-n matrix,
% y: k-by-1 vector, each element can be 1 or -1
% r: learning rate.
%
% Output:
% w: 1-by-n vector of the weight vector.
% theta: the theta weight parameter corresponding to w_0.

function [w,theta] = modified_perceptron_margin_train(x,y,r)
    [k,n] = size(x);
    %The weight vector is 'w'. 
    %Initializing the weight vector and theta to zeros.
    w = zeros(1,n);   
    theta = 0;
    %Find the proportion of examples marked with label '1' in y.
    positive = histc(y,1);
    negative = k - positive;
    ratio = positive/negative;
    margin_positive = 1;
    margin_negative = ratio*margin_positive;
    
    for iterate = 1:20
        for i = 1:k
            %Select the ith training example.
            training_example = x(i,:);
            product_i = dot(w,training_example);
            %If positive example, update by margin_positive.
            if(y(i) == 1)
                if(y(i)*(product_i + theta) <= margin_positive)
                    %Update the weight vector and theta.
                    for j = 1:n
                        w(j) = w(j) + r*y(i)*x(i,j);
                    end
                    theta = theta + r*y(i);
                end
            else
            %If negative example, update by margin_negative.
                if(y(i)*(product_i + theta) <= margin_negative)
                    %Update the weight vector and theta.
                    for j = 1:n
                        w(j) = w(j) + r*y(i)*x(i,j);
                    end
                    theta = theta + r*y(i);
                end
            end
        end
    end
        
