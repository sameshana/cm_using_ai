
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%             Signal framing (segmentation)            %
%              with MATLAB Implementation              %
%                                                      %
% Author: Ph.D. Eng. Hristo Zhivomirov        04/29/19 %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [FRMS, t] = framing(x, frlen, hop, fs)
% function: [FRMS, t] = framing(x, frlen, hop, fs)
%
% Input:
% x - whole signal in the time domain
% frlength - signal frame length
% hop - hop size
% fs - sampling frequency, Hz
%
% Output:
% FRMS - frame-matrix (time across columns, 
%                      indexes across rows)
% determination of the signal length 
xlen = length(x);
% matrix size estimation and preallocation
F = 1+fix((xlen-frlen)/hop);    % calculate the number of signal frames
FRMS = zeros(frlen, F);         % preallocate the frame-matrix
% framing
for f = 0:F-1
    % framing
    xframe = x(1+f*hop : frlen+f*hop);
    
    % update of the frame-matrix
    FRMS(:, 1+f) = xframe;
end
% calculation of the time vector
t = (frlen/2:hop:frlen/2+(F-1)*hop)/fs;
end