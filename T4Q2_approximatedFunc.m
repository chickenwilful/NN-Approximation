function o = T4Q2_approximatedFunc(x, V, W, numNeuron, a, b) 
    y = zeros(numNeuron+1);
    o = zeros(1);
    % Synaptic output of the hidden layer
    u = V * x;           
    % Output of the hidden layer
    for j = 1 : numNeuron 
       y(j) = a * (1 - exp(-b*u(j))) / (1 + exp(-b*u(j)));
    end
    y(numNeuron + 1) = -1;
    % Synaptic output of output layer
    s = W * y;
    % Output
    for j = 1 : 1 % only 1 neuron on output layer
       o(j) = a * (1 - exp(-b*s(j))) / (1 + exp(-b*s(j)));
    end    
end 
