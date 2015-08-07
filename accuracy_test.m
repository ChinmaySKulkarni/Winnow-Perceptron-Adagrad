% This function is used to find the accuracy of our algorithm over the 
% testing data.
%
% Input:
% x: k-by-n matrix,
% y: k-by-1 vector, each element can be 1 or -1
% w: The learned weight vectors.
% theta: The learned value of theta.
%
% Output:
% accuracy: The accuracy of our learned algorithm.

function [accuracy] = accuracy_test(x,y,w,theta)
    [k,n] = size(x);
    %Record the number of mistakes made by the algorithm.
    mistakes = 0;
    for i = 1:k
        %Select the kth training example.
        training_example = x(i,:);
        product_i = dot(w,training_example);
        if(y(i)*(product_i + theta) <= 0)
            %Increment the number of mistakes.
            mistakes = mistakes + 1;
        end
    end
    accuracy = (100*(k - mistakes))/k;
