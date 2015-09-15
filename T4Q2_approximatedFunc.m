function O = T4Q2_approximatedFunc(X, V, W, numNeuron, a, b) 
    X = vertcat(X, repmat(-1, 1, size(X, 2)));    
    % Synaptic output of the hidden layer
    U = V * X;           
    % Output of the hidden layer
    Y = a * (1 - exp(-b*U)) ./ (1 + exp(-b*U));
    Y = vertcat(Y, repmat(-1, 1, size(Y, 2)));    
    % Synaptic output of output layer
    S = W * Y;
    % Output
    O = a * (1 - exp(-b*S)) ./ (1 + exp(-b*S));    
end 
