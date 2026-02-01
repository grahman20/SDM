function [accuracy] = getAccuracy(actuals, predictions, H, nt, CV)
% function [accuracy] = getAccuracy(actuals, predictions, H, nt, CV)
% Input:
%   actuals         - actual labels of the test data (nt x 1)
%   predictions     - predicted labels of the test data (nt x H)
%   H               - number of hypothesis/classifiers  
%   nt              - size of the test data
%   CV              - list of labels 
%
% Output:
%   accuracy        - final accuracy

    predictedLabel=majorityVoting(predictions, H, nt, CV);   
    accuracy = numel(find(predictedLabel==actuals))/nt;
end

%% function to find majority voting
function [predictedLabel]=majorityVoting(predictions, H, nt, CV)
% function [predictedLabel]=majorityVoting(predictions, H, nt, CV)
% Input:
%   predictions     - predicted labels of the test data (nt x H)
%   H               - number of hypothesis/classifiers  
%   nt              - size of the test data
%   CV              - list of labels 
%
% Output:
%   predictedLabel  - final predicted labels of the test data (nt x 1)
    
    predictedLabel = zeros(nt,1);
    for row = 1:nt
        votes = zeros(1,length(CV)); 
        for col = 1:H %H numbers of different classifiers
            if ~isnan(predictions(row,col))
                votes(predictions(row,col)) = votes(predictions(row,col)) + 1;
            end
        end 
        [~,I] = max(votes);
        predictedLabel(row) = CV(I);
    end
end
