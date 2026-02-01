function [predictions] = multiClassSDM(Xs, Ys,target_data, test_data, param, options)
% function [predictions] = multiClassSDM(Xs, Ys,target_data, test_data, param)
% Input:
%   Xs             - source training features
%   Ys             - source labels
%   target_data    - target training features and labels
%   test_data      - target testing features and labels
%   param          - parameters for the SVM
%   options        - parameters for MMD and Manifold 
% Output:
%   predictions    - predicted labels of the test data (nt x H)


    %% Set paths for SVM
    addpath('.\utils');
    addpath('.\tools\libsvm-weights-3.17\matlab');  
    
    %% Find number of classes and hypothesis (one vs one approach)
    CV=unique(Ys);
    Nc=length(CV);
    H=Nc*(Nc-1)/2;
    nt=size(test_data.test_labels,1);
    
    %% 1-to-1 multiclass classification
    predictions=zeros(H,nt);
    round=0;
    for i=1:Nc-1
        fc=CV(i,1);
        for j=i+1: Nc
           sc=CV(j,1);
           round=round+1;
           % Finding positive and negative features 
           train_data1.train_features=transpose(Xs);
           train_data1.train_labels=Ys;
           train_data.train_features=train_data1.train_features(:,train_data1.train_labels==fc|train_data1.train_labels==sc);
           train_data.train_labels=train_data1.train_labels(train_data1.train_labels==fc|train_data1.train_labels==sc);
           train_data.train_labels(train_data.train_labels==fc)=1;
           train_data.train_labels(train_data.train_labels==sc)=-1;
           pos_features = train_data.train_features(:,train_data.train_labels==1);
           neg_features = train_data.train_features(:,train_data.train_labels==-1);%-1
           
           % main algorithm
           [model,kernel_param,training_features] = learnSDM(pos_features, neg_features, target_data.target_features, param, options);
           
           % prediction by a single classifier
           test_kernel = getKernel(test_data.test_features, training_features, kernel_param);
           ay      = full(model.sv_coef)*model.Label(1);
           idx     = full(model.SVs);
           b       = -(model.rho*model.Label(1));
           pred_y    = test_kernel(:, idx)*ay + b;    
           pred_y(pred_y>=0)=fc;
           pred_y(pred_y<0)=sc;
           predictions(round,:)=pred_y;
        end
    end
    %transform into index
    predictions=transpose(predictions);
end
