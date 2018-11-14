import pywt
import numpy as np
import matplotlib.pyplot as plt
import wave

w1 = wave.open('sample.wav','r')

signal = w1.readframes(-1)
signal = np.fromstring(signal, 'Int16')

left, right = signal[0::2], signal[1::2]

x = np.arange(512)
# y = np.sin(2*np.pi*x/32)
coef, freqs=pywt.cwt(left,np.arange(1,129),'gaus1')
# print(coef)
plt.matshow(coef)
plt.show()

w1.close()
