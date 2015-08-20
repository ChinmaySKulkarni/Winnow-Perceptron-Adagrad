% This function implements the perceptron with margin algorithm.  
% It is run over different training sets and it returns the vector representing the number of mistakes
% made by the algorithm.
%
% Input:
% x: k-by-n matrix,
% y: k-by-1 vector, each element can be 1 or -1
% r: learning rate.
%
% Output:
% mistake_vector: The vector representing mistakes over every 500 examples.

function [mistake_vector] = perceptron_margin_find_mistakes(x,y,r)
    [k,n] = size(x);
    %The weight vector is 'w'. 
    %Initializing the weight vector and theta to zeros.
    w = zeros(1,n);   
    theta = 0;
    mistakes = 0;
    mistake_vector = zeros(1,(k/500));

    for i = 1:k
        %Select the ith training example.
        training_example = x(i,:);
        product_i = dot(w,training_example);
        if(mod(i,500) == 0)
            %Record the mistakes made for this group of 500 examples.
            mistake_vector(1,i/500) = mistakes;
        end
        if(y(i)*(product_i + theta) <= 0)
            %Increment the mistakes counter.
            mistakes = mistakes + 1;
        end
        if(y(i)*(product_i + theta) <= 1)
            %Update the weight vector and theta.
            for j = 1:n
                w(j) = w(j) + r*y(i)*x(i,j);
            end
            theta = theta + r*y(i);
        end
    end
    
