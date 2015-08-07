% Q.3 Online Learning as Batch Learning Algorthms.
% Run this file to perform the following tasks:
% Train all 5 algorithms for m=100,500,1000 and l=10, n=1000 (20 times each) over 10% of the entire data for that value of 'm'.
% Find accuracies in all cases of 'm' for all the algorithms by testing over a different 10% of the data set.
% Find optimal parameters after parameter tuning.
% Use these optimal parameters obtained to train over 100% of the noisy
% training data (20 iterations) and evaluate the model obtained on the
% clean test data to report the accuracy for each different 'm' value for
% every algorithm

l = 10;
n = 1000;
m = [100;500;1000];
% Initialize all the variables we will need.
% The optimal parameters are stored as a row vector where each element
% corresponds to the same-index value of m in the m vector i.e. [100;500;1000].
% Thus the optimal parameter 'p' corresponding to m = 1000 would be 
% p(1,3).
perceptron_margin_max_acc = zeros(1,numel(m));
perceptron_margin_learning_rate = zeros(1,numel(m));

winnow_max_acc = zeros(1,numel(m));
winnow_prom_demot_par = zeros(1,numel(m));

winnow_margin_max_acc = zeros(1,numel(m));
winnow_margin_prom_demot_par = zeros(1,numel(m));
winnow_margin_margin = zeros(1,numel(m));

adagrad_max_acc = zeros(1,numel(m));
adagrad_learning_rate = zeros(1,numel(m));

for m_index = 1:numel(m)
    [train_y,train_x] = gen(l,m(m_index),n,50000,1);
    %10% data is training set.
    d_1_x = train_x(1:5000,:);
    d_1_y = train_y(1:5000,:);
    %10% data is testing set.
    d_2_x = train_x(5001:10000,:);
    d_2_y = train_y(5001:10000,:);

    % For perceptron with margin (Tune Learning Rate):
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
    
    % For Winnow (Tune the promotion/demotion parameter):
    prom_demot_parameters = [1.1, 1.01, 1.005, 1.0005, 1.0001];    
    max_accuracy = -1;
    prom_demot_val = 0;
    for idx = 1:numel(prom_demot_parameters)
        [weights] = winnow_train(d_1_x,d_1_y,prom_demot_parameters(idx));
        accuracy_pc = accuracy_test(d_2_x,d_2_y,weights,-1*n);
        if(accuracy_pc > max_accuracy)
            max_accuracy = accuracy_pc;
            prom_demot_val = prom_demot_parameters(idx);
        end
    end
    winnow_max_acc(1,m_index) = max_accuracy;
    winnow_prom_demot_par(1,m_index) = prom_demot_val;
    
    % For Winnow with Margin (Tune combinations of the promotion/demotion parameters and the margin:
    margins = [2.0, 0.3, 0.04, 0.006, 0.001];
    prom_demot_parameters = [1.1, 1.01, 1.005, 1.0005, 1.0001];    
    max_accuracy = -1;
    prom_demot_val = 0;
    margin_val = 0;
    for idx = 1:numel(prom_demot_parameters)
        for idy = 1:numel(margins)
            [weights] = winnow_margin_train(d_1_x,d_1_y,prom_demot_parameters(idx),margins(idy));
            accuracy_pc = accuracy_test(d_2_x,d_2_y,weights,-1*n);
            if(accuracy_pc > max_accuracy)
                max_accuracy = accuracy_pc;
                prom_demot_val = prom_demot_parameters(idx);
                margin_val = margins(idy);
            end 
        end
    end  
    winnow_margin_max_acc(1,m_index) = max_accuracy;
    winnow_margin_prom_demot_par(1,m_index) = prom_demot_val;
    winnow_margin_margin(1,m_index) = margin_val;
    
    % For Adagrad. (Tune the learning rate.):
    rates = [1.5, 0.25, 0.03, 0.005, 0.001];
    max_accuracy = -1;
    learning_rate = 0;
    for idx = 1:numel(rates)
        [weights,theta] = adagrad_train(d_1_x,d_1_y,rates(idx));
        accuracy_pc = accuracy_test(d_2_x,d_2_y,weights,theta);
        if(accuracy_pc > max_accuracy)
            max_accuracy = accuracy_pc;
            learning_rate = rates(idx);
        end
    end
    adagrad_max_acc(1,m_index) = max_accuracy;
    adagrad_learning_rate(1,m_index) = learning_rate;
end

% for disp_index = 1:numel(m)
%     disp(sprintf('m = %d\n',m(disp_index)))
%     disp('For Perceptron With Margin:')
%     disp('Max accuracy:')
%     disp(perceptron_margin_max_acc(1,disp_index))
%     disp('Learning Rate:')
%     disp(perceptron_margin_learning_rate(1,disp_index))
%     
%     disp('For Winnow:')
%     disp('Max accuracy:')
%     disp(winnow_max_acc(1,disp_index))
%     disp('Promotion/Demotion Parameter:')
%     disp(winnow_prom_demot_par(1,disp_index))
%     
%     disp('For Winnow with Margin:')
%     disp('Max accuracy:')
%     disp(winnow_margin_max_acc(1,disp_index))
%     disp('Promotion/Demotion Parameter:')
%     disp(winnow_margin_prom_demot_par(1,disp_index))
%     disp('Margin:')
%     disp(winnow_margin_margin(1,disp_index))
%     
%     disp('For Adagrad:')
%     disp('Max accuracy:')
%     disp(adagrad_max_acc(1,disp_index))
%     disp('Learning Rate:')
%     disp(adagrad_learning_rate(1,disp_index))
% end

for m_index = 1:numel(m)
    [train_y,train_x] = gen(l,m(m_index),n,50000,1);
    [test_y,test_x] = gen(l,m(m_index),n,10000,0);

    % For perceptron.
    [weights,theta] = perceptron_train(train_x,train_y);
    accuracy_perceptron(m_index) = accuracy_test(test_x,test_y,weights,theta);
    
    % For perceptron with margin (With the best learning rate):
    [weights,theta] = perceptron_margin_train(train_x,train_y,perceptron_margin_learning_rate(1,m_index));
    accuracy_perceptron_margin(m_index) = accuracy_test(test_x,test_y,weights,theta);
           
    % For Winnow (With the best promotion/demotion parameter):
    [weights] = winnow_train(train_x,train_y,winnow_prom_demot_par(1,m_index));
    accuracy_winnow(m_index) = accuracy_test(test_x,test_y,weights,-1*n);
        
    % For Winnow with Margin (With the best combination of promotion/demotion parameters and the margin:
    [weights] = winnow_margin_train(train_x,train_y,winnow_margin_prom_demot_par(1,m_index),winnow_margin_margin(1,m_index));
    accuracy_winnow_margin(m_index) = accuracy_test(test_x,test_y,weights,-1*n);
        
    % For Adagrad. (With the best learning rate.):
    [weights,theta] = adagrad_train(train_x,train_y,adagrad_learning_rate(1,m_index));
    accuracy_adagrad(m_index) = accuracy_test(test_x,test_y,weights,theta);
end

for disp_index = 1:numel(m)
    disp(sprintf('m = %d\n',m(disp_index)))
    disp('For Perceptron:')
    disp('Accuracy:')
    disp(accuracy_perceptron(disp_index))
    
    disp('For Perceptron With Margin:')
    disp('Accuracy:')
    disp(accuracy_perceptron_margin(disp_index))
    disp('Learning Rate:')
    disp(perceptron_margin_learning_rate(1,disp_index))
    
    disp('For Winnow:')
    disp('Accuracy:')
    disp(accuracy_winnow(disp_index))
    disp('Promotion/Demotion Parameter:')
    disp(winnow_prom_demot_par(1,disp_index))
    
    disp('For Winnow with Margin:')
    disp('Accuracy:')
    disp(accuracy_winnow_margin(disp_index))
    disp('Promotion/Demotion Parameter:')
    disp(winnow_margin_prom_demot_par(1,disp_index))
    disp('Margin:')
    disp(winnow_margin_margin(1,disp_index))
    
    disp('For Adagrad:')
    disp('Accuracy:')
    disp(accuracy_adagrad(disp_index))
    disp('Learning Rate:')
    disp(adagrad_learning_rate(1,disp_index))
end