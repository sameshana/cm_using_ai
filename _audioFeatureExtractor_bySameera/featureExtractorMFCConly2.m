%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%             MFCC and Stat Feature Extractor          %
%              with MATLAB Implementation              %
%                                                      %
% Author: Sameera Darshana        18/09/2020           %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [features, labels, numVectorsPerFile] = featureExtractorMFCConly2(ads, ds, frLen, hop)
    k=0;
    features = []; labels = [];numVectorsPerFile = [];
    while hasdata(ads)
        [audioIn,dsInfo] = read(ads);
        audioIn = audioIn/max(abs(audioIn));           % normalize the signal
%         aFE = audioFeatureExtractor("SampleRate",ds, ...
%             "Window",hamming(frLen,"periodic"), ...
%             "OverlapLength",hop, ...              
%             "SpectralDescriptorInput","melSpectrum", ...
%             "spectralCentroid",true, ...
%             "spectralCrest",true, ...
%             "spectralDecrease",true, ...
%             "spectralEntropy",true, ...
%             "spectralFlatness",true, ...
%             "spectralFlux",true, ...
%             "spectralKurtosis",true, ...
%             "spectralRolloffPoint",true, ...
%             "spectralSkewness",true, ...
%             "spectralSlope",true, ...
%             "spectralSpread",true, ...
%             "harmonicRatio",true, ... 
%             "pitch",true);
%         %"mfcc",true, ...
%         %setExtractorParams(aFE,"mfcc","NumCoeffs",20)
%         mfccFeat = extract(aFE,audioIn);        
        
        aFE = audioFeatureExtractor("SampleRate",ds, ...
            "Window",hamming(frLen,"periodic"), ...
            "OverlapLength",hop, ...              
            "SpectralDescriptorInput","melSpectrum", ...
            "mfcc",true)
        setExtractorParams(aFE,"mfcc","NumCoeffs",13)
        mfccFeat = extract(aFE,audioIn);  

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

        feat = [mfccFeat, statFeat];     
        %feat = statFeat;

        numVec = size(feat,1);
        label = repelem(dsInfo.Label,numVec);

        numVectorsPerFile = [numVectorsPerFile,numVec];
        features = [features;feat];
        labels = [labels,label];
    end