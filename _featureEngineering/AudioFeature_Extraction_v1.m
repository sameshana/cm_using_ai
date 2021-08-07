clear all, close all, clc
dataDir = "../audioData/dataTest";

ads = audioDatastore(dataDir,'IncludeSubfolders',true, ...
    'FileExtensions','.wav', ...
    'LabelSource','foldernames')

%[adsTrain, adsTest] = splitEachLabel(ads,0.8);
%trainDatastoreCount = countEachLabel(adsTrain);
ds = 24000;                                    %fs = dsInfo.SampleRate;
frLen = (round(0.05*ds));                      % frame length
hop = round(frLen/2);                          % hop size

[features, labels, ~] = featureExtractor(ads, ds, frLen, hop);

M = mean(features,1);
S = std(features,[],1);
features = (features-M)./S;

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
%=======To a external file    ============================================%
labels = transpose(cellstr(labels));
features = array2table(features);
features.labels = labels;
writetable(features, 'features.csv');
%==========================================================================%
%==========================================================================%

dataDir = "C:\del\data\noise removal\OR1H"
adsTest = audioDatastore(dataDir,'IncludeSubfolders',true, ...
    'FileExtensions','.wav', ...
    'LabelSource','foldernames')

[features_test, ~, numVectorsPerFile] = featureExtractor(adsTest, ds, frLen, hop);
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
