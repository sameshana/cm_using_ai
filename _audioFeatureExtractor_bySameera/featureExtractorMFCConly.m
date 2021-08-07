%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%             MFCC and Stat Feature Extractor          %
%              with MATLAB Implementation              %
%                                                      %
% Author: Sameera Darshana        18/09/2020           %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [features, labels, numVectorsPerFile] = featureExtractorMFCConly(ads, ds, frLen, hop)
    k=0
    features = []; labels = [];numVectorsPerFile = [];
    while hasdata(ads)
        [audioIn,dsInfo] = read(ads);

        audioIn = audioIn/max(abs(audioIn));           % normalize the signal        
        coeffs = mfcc(audioIn,ds,"OverlapLength",hop,"Window",hamming(frLen,"periodic"),"NumCoeffs",12);       
        
        numVec = size(coeffs,1);
        label = repelem(dsInfo.Label,numVec);
        
        numVectorsPerFile = [numVectorsPerFile,numVec];
        features = [features;coeffs];
        labels = [labels,label];        
    end