clear;clc;

%% set inputs 

basepath='.\data\';
% basepath='C:\Users\grahman\CSU-MSH\MGR\Research\Experiment\TL\Toy\Run-1\SDM\';
sf='source.arff';
tf='target.arff';
srcFile=[basepath sf];
tgtFile=[basepath tf];
testFile=[basepath 'test.arff'];

%% build model and calculate accuracy
[Acc] = SDM(srcFile,tgtFile,testFile);
accuracy= Acc * 100;
fprintf('%s,%s,%.2f%%\n', sf, tf,accuracy);