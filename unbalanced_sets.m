% Q.4 Bonus Question.
% Run this file to perform the following tasks:
% Train the original perceptron with margin algorithm and the modified perceptron with margin algorithm
% for m=100,500,1000 and l=10, n=1000 (20 times) over 10% of the entire unbalanced data for that value of 'm'.
% Find accuracies in all cases of 'm' for both the algorithms by testing over a different 10% of the data set.
% Find optimal parameters after parameter tuning.
%
% Use these optimal parameters obtained to train over 100% of the
% unbalanced training data (20 iterations) and 
% evaluate the model obtained on the unbalanced test data to report the accuracy 
% for each different 'm' value for both algorithms.

l = 10;
n = 1000;
m = [100;500;1000];

% Initialize all the variables we will need.
% The optimal parameters are stored as a row vector where each element
% corresponds to the same-index value of m in the m vector i.e. [100;500;1000].
% Thus the optimal parameter 'p' corresponding to m = 1000 would be 
% p(1,3).
modfied_perceptron_margin_max_acc = zeros(1,numel(m));
perceptron_margin_max_acc = zeros(1,numel(m));

modified_perceptron_margin_learning_rate = zeros(1,numel(m));
perceptron_margin_learning_rate = zeros(1,numel(m));

modified_perceptron_margin_iter_vals = zeros(1,numel(m));

for m_index = 1:numel(m)
    [train_y,train_x] = unba_gen(l,m(m_index),n,50000,0.1);
    %10% data is training set.
    d_1_x = train_x(1:5000,:);
    d_1_y = train_y(1:5000,:);
    %10% data is testing set.
    d_2_x = train_x(5001:10000,:);
    d_2_y = train_y(5001:10000,:);

    % For Modified perceptron with margin (Tune Learning Rate):
    rates = [1.5, 0.25, 0.03, 0.005, 0.001];
    max_accuracy = -1;
    learning_rate = 0;
    iter_val = 0;
    for idx = 1:numel(rates)
        [weights,theta] = modified_perceptron_margin_train(d_1_x,d_1_y,rates(idx));
        accuracy_pc = accuracy_test(d_2_x,d_2_y,weights,theta);
        if(accuracy_pc > max_accuracy)
            max_accuracy = accuracy_pc;
            learning_rate = rates(idx);
        end
    end
    modfied_perceptron_margin_max_acc(1,m_index) = max_accuracy;
    modified_perceptron_margin_learning_rate(1,m_index) = learning_rate;
        
    % For original perceptron with margin (Tune Learning Rate):
    rates = [1.5, 0.25, 0.03, 0.005, 0.001];
    max_accuracy = -1;
    learning_rate = 0;
    for idx = 1:numel(rates)
        [weights,theta] = perceptron_margin_train(d_1_x,d_1_y,rates(idx));
        accuracy_pc = accuracy_test(d_2_x,d_2_y,weights,theta);
        if(accuracy_pc > max_accuracy)
            max_accuracy = accuracy_pc;
            learning_rate = rates(idx);
        end
    end
    perceptron_margin_max_acc(1,m_index) = max_accuracy;
    perceptron_margin_learning_rate(1,m_index) = learning_rate;
end

for m_index = 1:numel(m)
    [train_y,train_x] = unba_gen(l,m(m_index),n,50000,0.1);
    [test_y,test_x] = unba_gen(l,m(m_index),n,10000,0.1);

    % For Modified perceptron with margin (With the best learning rate):
    [weights,theta] = modified_perceptron_margin_train(train_x,train_y,modified_perceptron_margin_learning_rate(1,m_index));
    accuracy_modified_perceptron_margin(m_index) = accuracy_test(test_x,test_y,weights,theta);
    
    % For the original perceptron with margin (With the best learning rate):
    [weights,theta] = perceptron_margin_train(train_x,train_y,perceptron_margin_learning_rate(1,m_index));
    accuracy_perceptron_margin(m_index) = accuracy_test(test_x,test_y,weights,theta);
end

for disp_index = 1:numel(m)
    disp(sprintf('m = %d\n',m(disp_index)))
    disp('For Modified Perceptron:')
    disp('Modified Perceptron Accuracy:')
    disp(accuracy_modified_perceptron_margin(disp_index))
    disp('Modified Perceptron Learning Rate:')
    disp(modified_perceptron_margin_learning_rate(1,disp_index))
        
    disp('For the Original Perceptron With Margin:')
    disp('Original Perceptron Accuracy:')
    disp(accuracy_perceptron_margin(disp_index))
    disp('Original Learning Rate:')
    disp(perceptron_margin_learning_rate(1,disp_index))
end