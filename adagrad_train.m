% This function implements the adagrad algorithm on the training data.
%
% Input:
% x: k-by-n matrix,
% y: k-by-1 vector, each element can be 1 or -1
% r: learning rate.
%
% Output:
% w: 1-by-n vector of the weight vector.
% theta: the theta weight parameter corresponding to w_0.

function [w,theta] = adagrad_train(x,y,r)
    [k,n] = size(x);
    %The weight vector is 'w'. 
    %Initializing the weight vector and theta to zeros.
    w = zeros(1,n);   
    theta = 0;
    g = zeros(k,n+1);
    sum_g_squares = zeros(1,n+1);
    
    for iterate = 1:20
        for i = 1:k
            %Select the ith training example.
            training_example = x(i,:);
            product_i = dot(w,training_example);
            %Calculate the value of g for this example.
            if(y(i)*(product_i + theta) <= 1)
                for j = 1:n
                    g(i,j) = -1*y(i)*x(i,j);
                end
                %Also update g for theta.
                g(i,n+1) = -1*y(i);
            else
                for index = 1:n+1
                    g(i,index) = 0;
                end
            end   
            %Calculate the value of sum of gradients' squares.

            for j = 1:n+1
                sum_g_squares(j) = sum_g_squares(j) + g(i,j).^2;
            end

            if(y(i)*(product_i + theta) <= 1)
                %Update the weight vector and theta.    
                for j = 1:n
                    if (sum_g_squares(j) ~= 0)
                        w(j) = w(j) + r*y(i)*x(i,j)/(sqrt(sum_g_squares(j)));
                    end
                end
                %Also update for theta.
                theta = theta + r*y(i)*1/sqrt(sum_g_squares(n+1));
            end
        end
    end