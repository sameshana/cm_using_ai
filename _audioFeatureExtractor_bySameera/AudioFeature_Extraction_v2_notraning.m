%clear all, close all, clc F1 Ball, F2 Cage, F3 OuterRacer, F4
%Contamination

dataDir = "C:\ActualData\Final\All_24KHz_1s_Final1000each";
dataDir = "C:\_RUL\3"

ads = audioDatastore(dataDir,'IncludeSubfolders',true, ...
    'FileExtensions','.wav', ...
    'LabelSource','foldernames');

[adsTrain, adsTest] = splitEachLabel(ads,0.9);
%trainDatastoreCount = countEachLabel(adsTrain);
ds = 24000;                                    %fs = dsInfo.SampleRate;
frLen = 1200 %(round(0.03*ds));                % frame length
hop = round(frLen/2);                          % hop size

[features, labels, ~] = featureExtractorMFCConly2(ads, ds, frLen, hop);

% M = mean(features,1);
% S = std(features,[],1);
% features = (features-M)./S;

%=======To a external file    ============================================%
labels = transpose(cellstr(labels));
featuresTable = array2table(features);
featuresTable.labels = labels;
writetable(featuresTable, 'features_RUL_5_all_5Feats.csv');
%==========================================================================%
