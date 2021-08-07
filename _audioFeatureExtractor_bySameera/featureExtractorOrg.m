%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%             MFCC and Stat Feature Extractor          %
%              with MATLAB Implementation              %
%                                                      %
% Author: Sameera Darshana        18/09/2020           %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [features, labels, numVectorsPerFile] = featureExtractorOrg(ads, ds, frLen, hop)

    features = []; labels = [];numVectorsPerFile = [];
    while hasdata(ads)
        [audioIn,dsInfo] = read(ads);
        %audioIn = audioIn/max(abs(audioIn));           % normalize the signal
      
        aFE = audioFeatureExtractor("SampleRate",ds, ...
            "Window",hamming(frLen,"periodic"), ...
            "OverlapLength",hop, ...            
            "mfcc",true);
        feat = extract(aFE,audioIn);
        numVec = size(feat,1);
        label = repelem(dsInfo.Label,numVec);        
        numVectorsPerFile = [numVectorsPerFile,numVec];
        features = [features;feat];
        labels = [labels,label];
    end