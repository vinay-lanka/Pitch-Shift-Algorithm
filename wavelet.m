%PathOriginal=fullfile('C:\Users\Downloads','original.wav');
[input,fs] = audioread('original.wav');
input = input(:,1);
dt = 1/fs;
t = 0:dt:(length(input)*dt)-dt;
plot(t,input); xlabel('Seconds'); ylabel('Amplitude');
f = input(:,1);
c = 2; % pitch factor
wlet = 'morl';
coefs = cwtft(f, 'wavelet', wlet);
absc = abs(coefs.cfs).*c; %mag
phac = angle(coefs.cfs); %phase
phac = phac.* c;
phac_unwrap = unwrap(phac);
coefs.cfs = absc.*exp(1i*phac_unwrap); %coefs_shifted
coefs.scales = coefs.scales./c; %scales_shifted
f_shifted = icwtft(coefs);
result = f_shifted;
maxx = max(abs(f));
maxy = max(abs(result));
result = (result/maxy)*maxx;
audiowrite('new1.wav', result,fs);
[y, Fs] = audioread('new1.wav');
player = audioplayer(y, Fs);
play(player)
figure
plot(result)