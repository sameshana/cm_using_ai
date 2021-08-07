%clear all, close all, clc F1 Ball, F2 Cage, F3 OuterRacer, F4
%Contamination

dataDir = "C:\ActualData\Final\All_24KHz_1s_Final1000each";
dataDir = "C:\_RUL\4"

ads = audioDatastore(dataDir,'IncludeSubfolders',true, ...
    'FileExtensions','.wav', ...
    'LabelSource','foldernames');

[adsTrain, adsTest] = splitEachLabel(ads,0.9);
%trainDatastoreCount = countEachLabel(adsTrain);
ds = 24000;                                    %fs = dsInfo.SampleRate;
frLen = 1024 %(round(0.03*ds));                      % frame length
hop = round(frLen/2);                          % hop size

[features, labels, ~] = featureExtractorMFCConly2(ads, ds, frLen, hop);

M = mean(features,1);
S = std(features,[],1);
features = (features-M)./S;

%=======To a external file    ============================================%
labels = transpose(cellstr(labels));
featuresTable = array2table(features);
featuresTable.labels = labels;
writetable(featuresTable, 'features_13MFCC_5000.csv');
%==========================================================================%


trainedClassifier = fitcknn( ...
    features, ...
    labels, ...
    'Distance','euclidean', ...
    'NumNeighbors',5, ...
    'DistanceWeight','squaredinverse', ...
    'Standardize',false, ...
    'ClassNames',unique(labels));

k = 5;
c = cvpartition(labels,'KFold',k); % 5-fold stratified cross validation
partitionedModel = crossval(trainedClassifier,'CVPartition',c);

fprintf('=================================');
validationAccuracy = 1 - kfoldLoss(partitionedModel,'LossFun','ClassifError');
fprintf('\nValidation accuracy = %.2f%%\n', validationAccuracy*100);

validationPredictions = kfoldPredict(partitionedModel);
figure
cm = confusionchart(labels,validationPredictions,'title','Validation Accuracy(Per Frame)');
cm.ColumnSummary = 'column-normalized';
cm.RowSummary = 'row-normalized';

%==========================================================================%

% dataDir = "C:\del\data\noise removal\OR1H"
% adsTest = audioDatastore(dataDir,'IncludeSubfolders',true, ...
%     'FileExtensions','.wav', ...
%     'LabelSource','foldernames')

[features_test, actualLabels, numVectorsPerFile] = featureExtractorMFCConly(adsTest, ds, frLen, hop);
M = mean(features_test,1);
S = std(features_test,[],1);
stdTestfeatures = (features_test-M)./S;

prediction = predict(trainedClassifier,stdTestfeatures);
prediction = categorical(string(prediction));

r2 = prediction(1:numel(adsTest.Files));
idx = 1;
for ii = 1:numel(adsTest.Files)
    r2(ii) = mode(prediction(idx:idx+numVectorsPerFile(ii)-1));
    idx = idx + numVectorsPerFile(ii);
end

figure('Units','normalized','Position',[0.4 0.4 0.4 0.4])
cm = confusionchart(adsTest.Labels,r2,'title','Test Accuracy (Per File)');
cm.ColumnSummary = 'column-normalized';
cm.RowSummary = 'row-normalized';

figure('Units','normalized','Position',[0.4 0.4 0.4 0.4])
cm = confusionchart(actualLabels,prediction,'title','Test Accuracy(Per File)');
cm.ColumnSummary = 'column-normalized';
cm.RowSummary = 'row-normalized';
