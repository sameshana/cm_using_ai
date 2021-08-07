%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%             MFCC and Stat Feature Extractor          %
%              with MATLAB Implementation              %
%                                                      %
% Author: Sameera Darshana        18/09/2020           %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [features, labels, numVectorsPerFile] = featureExtractor(ads, ds, frLen, hop)

    features = []; labels = [];numVectorsPerFile = [];
    while hasdata(ads)
       [audioIn,dsInfo] = read(ads);

       bandpassFeat = []; 
       ft = abs(fft(audioIn));
       %ft = abs(pwelch(audioIn));
       
       freqBin = 1000;
       feat = [mean(ft)];
       for c = 0:7
            feat = [feat mean(ft(c*freqBin+1:(c+1)*freqBin))];
       end
   
        numVec = size(feat,1);
        label = repelem(dsInfo.Label,numVec);
        
        numVectorsPerFile = [numVectorsPerFile,numVec];
        features = [features;feat];
        labels = [labels,label];

    end

