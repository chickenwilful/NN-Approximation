function [V_opt, W_opt, eTrain, eValidate] = T4Q2_approximatorMLP(trainSet, validationSet, numNeuron, maxIter, a, b, alpha)
    % numNeuron: num. neurons on the hidden layer (only 1 hidden layer)
    % a, b: learning parameters for bipolar sigmoidal activation function 
    disp(['Train MLP with #Neuron = ' num2str(numNeuron)]);
    desiredTrain = 0.8 * sin(pi * trainSet);
    desiredValidation = 0.8 * sin(pi * validationSet);

    ePlotTrain = (1);
    ePlotTest = (1);

    % Add bias as 1 feature
    trainSet = vertcat(trainSet, repmat(-1, 1, size(trainSet, 2)));

    % Initialize weight
    V = rand(numNeuron, 1+1); % weight matrix of input/hiddenlayer
    W = rand(1, numNeuron+1); % weight matrix of of hidden/outputlayer 

    eValOld = -1; iter_opt = -1;    
    for t = 1 : maxIter
        e = 0;        
        for i = 1 : size(trainSet, 2) 
            x = trainSet(:, i);
            d = desiredTrain(:, i);
            % Synaptic output of the hidden layer
            u = V * x;           
            % Output of the hidden layer
            y = zeros(numNeuron + 1, 1);           
            y(1:numNeuron, :) = a * (1 - exp(-b*u)) ./ (1 + exp(-b*u));           
            y(numNeuron + 1) = -1;
            % Synaptic output of output layer
            s = W * y;
            % Output
            o = a * (1 - exp(-b*s)) ./ (1 + exp(-b*s));
            % Calculate train error
            e = e + (d - o).^2; % total error of the iteration
            % error signal of output layer: e(k) = (d(k) - o(k))f'(s(k))
            esig_o = (d - o) .* (0.5 * a * b * ( 1 - 1/a^2 * o.^2));
            % error signal of hidden layer e(j) = f'(u(j)) * (w'(j) * e_o)
            esig_y = (0.5 * a * b * ( 1 - 1/a^2 * y.^2)) .* (W' * esig_o); 
            % Update weights
            W = W + alpha * esig_o * y';
            V = V + alpha * esig_y(1:numNeuron, :) * x';                                   
        end

        % Calculate validation error
        eValidate = T4Q2_calcMSE(validationSet, desiredValidation, V, W, numNeuron, a, b);
        if (t > 800 && iter_opt == -1 && eValOld < eValidate) % stopping point to avoid overfit 
            W_opt = W;
            V_opt = V;
            iter_opt = t-1;
            break;
        end
        eValOld = eValidate;
        eTrain = e / size(trainSet, 2); % MSE
        ePlotTest(t) = eValidate;
        ePlotTrain(t) = eTrain;
    end

    if (iter_opt == -1)
        V_opt = V;
        W_opt = W;
        iter_opt = maxIter;
    end

    %% Plot results
    figure(numNeuron), plot(ePlotTrain, 'color', 'b'); hold on;
    plot(ePlotTest, 'color', 'r');
    line([iter_opt iter_opt],get(gca,'YLim'),'Color',[1 0 0])
    legend('Train Error', 'Test Error')
    title(strcat('Error Each Iteration at #Neuron=', num2str(numNeuron)));
    xlabel('Iteration')
    ylabel('Mean Square Error')
    hold off;    
end 