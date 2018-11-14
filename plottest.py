import matplotlib.pyplot as plt
import numpy as np
import wave
import sys


w1 = wave.open('sample.wav','r')

signal = w1.readframes(-1)
signal = np.fromstring(signal, 'Int16')

left, right = signal[0::2], signal[1::2]
lf, rf = np.fft.rfft(left), np.fft.rfft(right)
plt.figure(1)
plt.title('Input Signal Wave')
plt.plot(lf)
# plt.show()
# sys.exit(0)


w2 = wave.open('pitch2.wav','r')

newsignal = w2.readframes(-1)
newsignal = np.fromstring(newsignal, 'Int16')

left1, right1 = newsignal[0::2], newsignal[1::2]
lf1, rf1 = np.fft.rfft(left1), np.fft.rfft(right1)
plt.figure(2)
plt.title('Pitch Shifted Signal')
plt.plot(lf1)
plt.show()
sys.exit(0)
