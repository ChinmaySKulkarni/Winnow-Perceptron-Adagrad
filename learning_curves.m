% Q.2 Run this file to perform the following tasks:
% Train all 5 algorithms for n=40,80,120,160 and n=200 (20 times each), l=10, m=20 over 10% of the entire data for that value of 'n'.
% Find accuracies in all cases of 'n' for all the algorithms by testing over a different 10% of the data set.
% Find optimal parameters after parameter tuning.
% Find the number of mistakes made for these optimal parameters for each
% algorithm under n=40,80,120,160 and n=200 circumstances.
% In this, we plot the graph of W vs. n for all algorithms with optimal parameters.
% W is the number of mistakes made by the algorithm before it stops making
% mistakes for a coninuous interval of 'R' (convergence criteria) examples.

l = 10;
m = 20;    
n = 40:40:200;

% Initialize all the variables we will need.
% The optimal parameters are stored as a row vector where each element
% corresponds to the same-index value of n in the n vector.
% Thus the optimal parameter 'p' corresponding to n = 80 would be 
% p(1,2).
perceptron_margin_max_acc = zeros(1,numel(n));
perceptron_margin_learning_rate = zeros(1,numel(n));

winnow_max_acc = zeros(1,numel(n));
winnow_prom_demot_par = zeros(1,numel(n));

winnow_margin_max_acc = zeros(1,numel(n));
winnow_margin_prom_demot_par = zeros(1,numel(n));
winnow_margin_margin = zeros(1,numel(n));

adagrad_max_acc = zeros(1,numel(n));
adagrad_learning_rate = zeros(1,numel(n));

for n_index = 1:numel(n)
    [y,x] = gen(l,m,n(n_index),50000,0);
    %10% data is training set.
    d_1_x = x(1:5000,:);
    d_1_y = y(1:5000,:);
    %10% data is testing set.
    d_2_x = x(5001:10000,:);
    d_2_y = y(5001:10000,:);

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
    perceptron_margin_max_acc(1,n_index) = max_accuracy;
    perceptron_margin_learning_rate(1,n_index) = learning_rate;
    
    % For Winnow (Tune the promotion/demotion parameter):
    prom_demot_parameters = [1.1, 1.01, 1.005, 1.0005, 1.0001];    
    max_accuracy = -1;
    prom_demot_val = 0;
    for idx = 1:numel(prom_demot_parameters)
        [weights] = winnow_train(d_1_x,d_1_y,prom_demot_parameters(idx));
        accuracy_pc = accuracy_test(d_2_x,d_2_y,weights,-1*n(n_index));
        if(accuracy_pc > max_accuracy)
            max_accuracy = accuracy_pc;
            prom_demot_val = prom_demot_parameters(idx);
        end
    end
    winnow_max_acc(1,n_index) = max_accuracy;
    winnow_prom_demot_par(1,n_index) = prom_demot_val;
    
    % For Winnow with Margin (Tune combinations of the promotion/demotion parameters and the margin:
    margins = [2.0, 0.3, 0.04, 0.006, 0.001];
    prom_demot_parameters = [1.1, 1.01, 1.005, 1.0005, 1.0001];    
    max_accuracy = -1;
    prom_demot_val = 0;
    margin_val = 0;
    for idx = 1:numel(prom_demot_parameters)
        for idy = 1:numel(margins)
            [weights] = winnow_margin_train(d_1_x,d_1_y,prom_demot_parameters(idx),margins(idy));
            accuracy_pc = accuracy_test(d_2_x,d_2_y,weights,-1*n(n_index));
            if(accuracy_pc > max_accuracy)
                max_accuracy = accuracy_pc;
                prom_demot_val = prom_demot_parameters(idx);
                margin_val = margins(idy);
            end 
        end
    end  
    winnow_margin_max_acc(1,n_index) = max_accuracy;
    winnow_margin_prom_demot_par(1,n_index) = prom_demot_val;
    winnow_margin_margin(1,n_index) = margin_val;
    
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
    adagrad_max_acc(1,n_index) = max_accuracy;
    adagrad_learning_rate(1,n_index) = learning_rate;
end


for disp_index = 1:numel(n)
    disp(sprintf('n = %d\n',n(disp_index)))
    disp('For Perceptron With Margin:')
    disp('Max accuracy:')
    disp(perceptron_margin_max_acc(1,disp_index))
    disp('Learning Rate:')
    disp(perceptron_margin_learning_rate(1,disp_index))
    
    disp('For Winnow:')
    disp('Max accuracy:')
    disp(winnow_max_acc(1,disp_index))
    disp('Promotion/Demotion Parameter:')
    disp(winnow_prom_demot_par(1,disp_index))
    
    disp('For Winnow with Margin:')
    disp('Max accuracy:')
    disp(winnow_margin_max_acc(1,disp_index))
    disp('Promotion/Demotion Parameter:')
    disp(winnow_margin_prom_demot_par(1,disp_index))
    disp('Margin:')
    disp(winnow_margin_margin(1,disp_index))
    
    disp('For Adagrad:')
    disp('Max accuracy:')
    disp(adagrad_max_acc(1,disp_index))
    disp('Learning Rate:')
    disp(adagrad_learning_rate(1,disp_index))
end

for test_index = 1:numel(n)
        
    [y,x] = gen(l,m,n(test_index),50000,0);
    total_mistakes_perceptron(test_index) = perceptron_find_mistakes_convergence(x,y);
    total_mistakes_perceptron_margin(test_index) = perceptron_margin_find_mistakes_convergence(x,y,perceptron_margin_learning_rate(1,test_index));
    total_mistakes_winnow(test_index) = winnow_find_mistakes_convergence(x,y,winnow_prom_demot_par(1,test_index));
    total_mistakes_winnow_margin(test_index) = winnow_margin_find_mistakes_convergence(x,y,winnow_margin_prom_demot_par(1,test_index), winnow_margin_margin(1,test_index));
    total_mistakes_adagrad(test_index) = adagrad_find_mistakes_convergence(x,y,adagrad_learning_rate(1,test_index));

    disp(sprintf('n = %d\n',n(test_index)))
    disp(sprintf('Total Mistakes Perceptron = \t%d',total_mistakes_perceptron(test_index)))
    disp(sprintf('Total Mistakes Perceptron With Margin = \t%d', total_mistakes_perceptron_margin(test_index)))
    disp(sprintf('Total Mistakes Winnow = \t%d',total_mistakes_winnow(test_index)))
    disp(sprintf('Total Mistakes Winnow With Margin = \t%d', total_mistakes_winnow_margin(test_index)))
    disp(sprintf('Total Mistakes Adagrad = \t%d',total_mistakes_adagrad(test_index)))
end

%Plot for each algorithm W vs. n
x = [0 n];
yp = [0 total_mistakes_perceptron];
ypm = [0 total_mistakes_perceptron_margin];
yw = [0 total_mistakes_winnow];
ywm = [0 total_mistakes_winnow_margin];
ya = [0 total_mistakes_adagrad];
   
figure;
plot(x,yp,'r',x,ypm,'g',x,yw,'b',x,ywm,'y',x,ya,'k')
xlabel('Values n');
ylabel('Number of Mistakes W');
title('W vs. n ');
legend('Perceptron','Perceptron With Margin','Winnow','Winnow With Margin','Adagrad');
