[data,fs] = audioread('C:\__001s\00000.wav');   

data_fft = fft(data);
plot(abs(data_fft(:,1)));
title('1khz-fft');