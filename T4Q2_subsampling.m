function [training, validate] = T4Q2_subsampling(dataset, numTrain, numTest) 
    dataset = dataset(:, randperm(size(dataset, 2)));
    % training <-- first numTrain columns
    training = dataset(1:numTrain);
    % validate <-- next numTest columns
    validate = dataset(numTrain+1:numTrain+numTest);
end
