% This function implements the winnow algorithm. It is run over different training sets and it
% returns the vector representing the number of mistakes.
%
% Input:
% x: k-by-n matrix,
% y: k-by-1 vector, each element can be 1 or -1
% param: The promotion/demotion parameter.
%
% Output:
% mistake_vector: The vector representing mistakes over every 500 examples.

function [mistake_vector] = winnow_find_mistakes(x,y,param)
    [k,n] = size(x);
    %The weight vector is 'w'. 
    %Initializing the weight vector to ones.
    w = ones(1,n);   
    theta = -n;
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
            %Update the weight vector.
            for j = 1:n
               w(j) = w(j)*(param.^(y(i) * x(i,j)));
            end
        end
    end
    
