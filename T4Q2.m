clc
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
[trainValidateSet, testSet] = T4Q2_subsampling(Dataset, 21, 0);

for i = 1:split
    disp(['Split #' num2str(i) ':']);
    % Training set & Validation set
    [trainSet, validationSet] = T4Q2_subsampling(trainValidateSet, 7, 7);
    
    for numNeuron = 1:MAXNEURONS
        [V, W, ETrain, EVal] = T4Q2_approximatorMLP(trainSet, validationSet, numNeuron, MAXEPOCHS, a, b, alpha);
        error(i,numNeuron) = EVal;
    end  
end
%% Plot MSE according to #Neuron
error
avgTestErr = mean(error)
figure(21), plot(avgTestErr); hold on;
    title('Graph of Avg Test Error for each model')
    xlabel('Number of Neurons')
    ylabel('Avg Test Error')
    hold off;

%% Optimal #Neuron    
[minErr, minErrIndex] = min(avgTestErr);
minErrIndex % Number of neurons for best model

%% Create final model (with optimal numNeuron)
numNeuron_opt = minErrIndex;
[trainSet, validationSet] = T4Q2_subsampling(trainValidateSet, 14, 7);
[V_opt, W_opt, ETrain, EVal] = T4Q2_approximatorMLP(trainSet, validationSet, numNeuron_opt, MAXEPOCHS, a, b, alpha);

disp('Final Model Configuration')
minErrIndex
V_opt
W_opt

%% True test error estimate 
% desiredResult = 0.8 * sin(pi * testSet);
% true_test_error = T4Q2_calcMSE(testSet, desiredResult, V_opt, W_opt, numNeuron_opt, a, b);
% true_test_error

% %% Plot approximated function on test set
% desire = (1);
% approximated = (1);
% data = testSet;
% data = sort(data);
% % for i = 1: size(data, 2)
%     desire = 0.8 * sin(pi * data);
%     approximated = T4Q2_approximatedFunc(data, V_opt, W_opt, numNeuron_opt, a, b);
% % end
% size(desire)
% size(approximated)
% figure(22), plot(data, desire, '-b', data, approximated, '-r'); hold on;
% legend('True value', 'Approximated value')
% title('Approximated Function on Testing')
% xlabel('x')
% hold off;
%     

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







