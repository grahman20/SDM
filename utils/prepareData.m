function [Xs, Ys,target_data, test_data] = prepareData(sourceFile,targetFile, testFile)
% function [accuracy] = mainSDM(sourceFile,targetFile, testFile)
% Input:
%   sourceFile     - source domain training data (e.g. source.arff)
%   targetFile     - target domain training data (e.g. target.arff)
%   testFile       - target domain testing data (e.g. test.arff)
%
% Output:
%   Xs             - source training features
%   Ys             - source labels
%   target_data    - target training features and labels
%   test_data      - target testing features and labels


    %% Set paths for Weka
    javaaddpath('.\tools\weka\weka.jar');
    
    %% Read source file
    [srcX,srcY]=readArff(sourceFile); 
    srcX = srcX ./ repmat(sum(srcX,2),1,size(srcX,2)); 
    Xs = srcX;    clear srcX
    Ys=srcY; clear srcY

    %% Read target file
    [tgtX,tgtY]=readArff(targetFile); 
    tgtX = tgtX ./ repmat(sum(tgtX,2),1,size(tgtX,2)); 
    Xt = tgtX;    clear tgtX
    Yt=tgtY; clear tgtY    
    target_data.target_features=transpose(Xt);
    target_data.target_labels=Yt;
    
     %% Read test file
    [testX,testY]=readArff(testFile);  
    testX = testX ./ repmat(sum(testX,2),1,size(testX,2)); 
    Xtest = testX;    clear testX
    Ytest=testY; clear testY
    test_data.test_features=transpose(Xtest);
    test_data.test_labels=Ytest;   
    
end


