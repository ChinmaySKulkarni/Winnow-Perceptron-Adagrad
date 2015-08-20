% This function implements the winnow algorithm. It is run over different training sets and it
% returns the total number of mistakes reached till the point when a
% continuous set of 'R' examples are found such that there are no mistakes
% on those examples.
%
% Input:
% x: k-by-n matrix,
% y: k-by-1 vector, each element can be 1 or -1
% param: The promotion/demotion parameter.
%
% Output:
% mistakes = Total number of mistakes made till the continuous sequence of 
% 'R' examples are found on which the algorithm makes no mistakes.

function [mistakes] = winnow_find_mistakes_convergence(x,y,param)
    [k,n] = size(x);
    %The weight vector is 'w'. 
    %Initializing the weight vector to ones.
    w = ones(1,n);   
    theta = -n;
    mistakes = 0;
    % Convergence criteria.
    R = 1000;
    unmistaken = 0;

    %Iterate over the entire examples a maximum of 10 times.
    for iterate = 1:10
        for i = 1:k
            if(unmistaken == R)
                break;
            end
            %Select the ith training example.
            training_example = x(i,:);
            product_i = dot(w,training_example);
            if(y(i)*(product_i + theta) <= 0)
                %Increment the mistakes counter.
                mistakes = mistakes + 1;
                unmistaken = 0;
                %Update the weight vector.
                for j = 1:n
                   w(j) = w(j)*(param.^(y(i) * x(i,j)));
                end
            else
                unmistaken = unmistaken + 1;
            end
        end
        if (unmistaken == R)
            break;
        else
            disp('unmistaken count < R! Iterating over all examples once more...')
        end    
    end
    if (unmistaken ~= R)
        disp('unmistaken count STILL < R!! Reduce R!')
    end
    