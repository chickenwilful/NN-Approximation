clc
% Figure 1 illustrates a MLP network design to approximate function
% f(x) = 0.8sin(? x) for ?1 ? x ?1.
% All neurons have bipolar sigmoidal activation function with a = 1.0 and b = 0.5.
% Determine the optimal design of the network in terms of the number of hidden
% neurons by using the following cross-validation procedures. Use learning factor = 0.6
% and maximum number of iterations = 5000. The dataset is created by choosing 21 data
% points uniformly covering the function.

MAXEPOCHS = 5000; % maximum number of iteration
MAXNEURONS = 10; % Set maximum number of neurons

% learning parameters for activation function: 
% f(u) = a(1 - exp(-b * u)) / (1 + exp(-b * u))
a = 1.0; 
b = 0.5; 

alpha = 0.6; % learning parameter for changing weight 

split = 3; % 3 data splits random subsampling
error = zeros(split,MAXNEURONS);

% Dataset 
Dataset = -1 : 1/10 : 1;

% Take out Test set, to preserve
[trainValidateSet, testSet] = T4Q2_subsampling(Dataset, 14, 7);

for i = 1:split
    % Training set & Validation set
    [trainSet, validationSet] = T4Q2_subsampling(trainValidateSet, 7, 7);

    for numNeuron = 1:MAXNEURONS
        [V, W, ETrain, EVal] = T4Q2_approximatorMLP(trainSet, validationSet, numNeuron, MAXEPOCHS, a, b, alpha);
        error(i,numNeuron) = EVal;
    end  
end

error
avgTestErr = mean(error)
figure(21), plot(avgTestErr); hold on;
    title('Graph of Avg Test Error for each model')
    xlabel('Number of Neurons')
    ylabel('Avg Test Error')
    hold off;
[minErr, minErrIndex] = min(avgTestErr);


minErrIndex % Number of neurons for best model

%% Create final model (with optimal numNeuron)
numNeuron_opt = minErrIndex;
% Holdout method (split 14 poinsts to training(10), validation(4))
[trainSet, validationSet] = T4Q2_subsampling(trainValidateSet, 10, 4);
[V_opt, W_opt, ETrain, EVal] = T4Q2_approximatorMLP(trainSet, validationSet, numNeuron_opt, MAXEPOCHS, a, b, alpha);

%% True test error estimate 
desiredResult = 0.8 * sin(pi * testSet);
true_test_error = T4Q2_calcMSE(testSet, desiredResult, V_opt, W_opt, numNeuron_opt, a, b);
true_test_error

%% Plot approximated function on test set
desire = (1);
approximated = (1);
data = testSet;
data = sort(data);
for i = 1: size(data, 2)
    desire(i) = 0.8 * sin(pi * data(i));
    approximated(i) = T4Q2_approximatedFunc(data(i), V_opt, W_opt, numNeuron_opt, a, b);
end
figure(22), plot(data, desire, '-b', data, approximated, '-r'); hold on;
legend('True value', 'Approximated value')
title('Approximated Function on Testing')
xlabel('x')
hold off;
    
%% Plot approximated function on the whole dataset
desire = (1);
approximated = (1);
data = Dataset;
data = sort(data);
for i = 1: size(data, 2)
    desire(i) = 0.8 * sin(pi * data(i));
    approximated(i) = T4Q2_approximatedFunc(data(i), V_opt, W_opt, numNeuron_opt, a, b);
end
figure(23), plot(data, desire, '-b', data, approximated, '-r'); hold on;
legend('True value', 'Approximated value')
title('Approximated Function on whole datas')
xlabel('x')
hold off;







