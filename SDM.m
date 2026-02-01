function [accuracy] = SDM(sourceFile,targetFile, testFile)
% function [accuracy] = SDM(sourceFile,targetFile, testFile)
% Input:
%   sourceFile     - source domain training data (e.g. source.arff)
%   targetFile     - target domain training data (e.g. target.arff)
%   testFile       - target domain testing data (e.g. test.arff)
%
% Output:
%   accuracy       - final accuracy
  
    %% Preparing data by reading *.arff files and converting into matlab format
    %% Set paths for utils files
    addpath('.\utils');
    fprintf('SDM starts....\nPreparing data...\n');
    [Xs, Ys,target_data, test_data] = prepareData(sourceFile,targetFile, testFile);
    nt=size(test_data.test_labels,1);
    
    %% Set parameters for SVM

    param.C = 10;
    param.Cu = 1; % Cu should be less than C
    param.Cu_max      = 10*param.Cu; % add at most rho patterns at each iteration
    param.rho         = 10;   
    param.max_iter    = 100;
    param.max_unl_num = 5;
    param.kernel_type ='gaussian'; % 'gaussian' or 'linear';
    
    %% Set optiones for MMD and Manifold
    dd=20;
    [numRows,numCols] = size(Xs);
    if dd>numRows
        dd=numRows-1;
    end   
    if dd>numCols
        dd=numCols-1;
    end
    [numRows,numCols] = size(target_data.target_features);
    if dd>numRows
        dd=numRows-1;
    end   
    if dd>numCols
        dd=numCols-1;
    end
    options.d = dd; %defualt 20
    options.rho = 1.0;
    options.p = 10;
    options.lambda = 10.0;
    options.eta = 0.1;
    options.T = 10;
    %% Find number of classes and hypothesis
    CV=unique(Ys);
    Nc=length(CV);
    H=Nc*(Nc-1)/2;
    
    %% 1-to-1 multiclass classification
    fprintf('Running multiClassSDM...\n');
    [predictions] = multiClassSDM(Xs, Ys,target_data, test_data, param, options);
   
    %% Calculate classification accuracy on the test data
    fprintf('Calculating classification accuracy...\n');
    [accuracy] = getAccuracy(test_data.test_labels, predictions, H, nt, CV);
end

