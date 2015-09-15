function E = T4Q2_calcMSE(X, D, V, W, numNeuron, a, b)
    O = T4Q2_approximatedFunc(X, V, W, numNeuron, a, b);
    E = sum((D - O).^2);
    E = E / size(X, 2);
end