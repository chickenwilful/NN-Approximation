function E = T4Q2_calcMSE(X, D, V, W, numNeuron, a, b)
    % Add bias as 1 feature
    X = vertcat(X, repmat(-1, 1, size(X, 2)));
    E = 0;
    for i = 1 : size(X, 2) 
        x = X(:, i);
        d = D(:, i);
        o = T4Q2_approximatedFunc(x, V, W, numNeuron, a, b);
        E = E + sum((d - o).^2); 
    end 
    E = E / size(X, 2);
end