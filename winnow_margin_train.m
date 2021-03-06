% This function implements the winnow with margin algorithm for the
% training data.
%
% Input:
% x: k-by-n matrix,
% y: k-by-1 vector, each element can be 1 or -1
% param: The promotion/demotion parameter.
% margin: the margin for which to run this algorithm.
%
% Output:
% w: 1-by-n vector of the weight vector.

function [w] = winnow_margin_train(x,y,param,margin)
    [k,n] = size(x);
    %The weight vector is 'w'. 
    %Initializing the weight vector to ones.
    w = ones(1,n);   
    theta = -n;
    
    for iterate = 1:20
        for i = 1:k
            %Select the ith training example.
            training_example = x(i,:);
            product_i = dot(w,training_example);
            if(y(i)*(product_i + theta) <= margin)
                %Update the weight vector.
                for j = 1:n
                   w(j) = w(j)*(param.^(y(i) * x(i,j)));
                end
            end
        end
    end