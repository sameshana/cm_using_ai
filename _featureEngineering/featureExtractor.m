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

        aFE = audioFeatureExtractor("SampleRate",ds, ...
            "Window",hamming(frLen,"periodic"), ...
            "OverlapLength",hop, ...
            "mfcc",true, ...
            "spectralSkewness",true, ...
            "spectralKurtosis",true);

        feat = extract(aFE,audioIn);

        audioIn = audioIn/max(abs(audioIn));           % normalize the signal
        N = length(audioIn);                           % signal length
        t = (0:N-1)/ds;                                % time vector

        [FRM, ~] = framing(audioIn, frLen, hop, ds);    % signal framing

        FRM = transpose(FRM);
        statFeat = [];
        for k = 1:size(FRM,1)
            frame = FRM(k,:);
            featRms = rms(frame);
            featStd = std(frame);
            [featEnvU, ~] = envelope((frame));
            featEnv = rms((featEnvU));
            featPsd = rms(pwelch(frame));
            featFft = rms(abs(fft(frame)));
            temp = [featRms, featStd, featEnv, featPsd, featFft];
            statFeat = [statFeat; temp];
        end

        feat = [feat, statFeat];        
        
        numVec = size(feat,1);
        label = repelem(dsInfo.Label,numVec);
        
        numVectorsPerFile = [numVectorsPerFile,numVec];
        features = [features;feat];
        labels = [labels,label];
    end