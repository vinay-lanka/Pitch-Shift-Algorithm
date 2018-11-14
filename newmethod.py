import sys
import numpy as np
from scipy.io import wavfile
from math import floor

# CONSTANTS
epsilon = sys.float_info.epsilon

class PhaseVocoder(object):
	"""docstring for PhaseVocoder"""
	def __init__(self, N=2**12, M=2**12, Rs=(2**12/8), w=np.hanning(2**12), alpha=1):
		super(PhaseVocoder, self).__init__()
		self.N	    = N		# FFT size
		self.M 	    = M		# Window size
		self.Rs 	= Rs 	# Synthesis hop size
		self.alpha  = alpha	# Timestretch factor
		self.w      = w 	# Analysis/Synthesis window

	def read_wav(self, filename):
		"""
		Read signal from .wav file
		filename: name of input .wav file
		returns fs: sampling frequency, x: signal stored in filename .wav file
		"""
		(fs, x) = wavfile.read(filename)
		if len(np.shape(x)) > 1:
			x = x[:,1]	# crudely convert from stereo to mono

		return (fs, x)

	def write_wav(self, filename, fs, x):
		"""
		Write signal to .wav file
		filename: name of output .wav file, fs: samling frequency, x: input signal
		"""
		wavfile.write(filename, fs,  np.array( x , dtype='int16'))

	def speedx(self, sound_array, factor):
	    """ Multiplies the sound's speed by some `factor` """
	    indices = np.round( np.arange(0, len(sound_array), factor) )
	    indices = indices[indices < len(sound_array)].astype(int)
	    return sound_array[ indices.astype(int) ]

	def pitchshift(self, snd_array, n):
	    """ Changes the pitch of a sound by ``n`` semitones. """
	    factor = 2**(1.0 * n / 12.0)
	    stretched = self.timestretch(snd_array, 1.0/factor)
	    return self.speedx(stretched, factor)


	def timestretch(self, x, alpha):
		"""
		Perform timestretch of a factor alpha to signal x
		x: input signal, alpha: timestrech factor
		returns: a signal of length T*alpha
		"""

		# Analysis/Synthesis window function
		w = self.w; N = self.N; M = self.M
		hM1 = int(floor((M-1)/2.))
		hM2 = int(floor(M/2.))

		# Synthesis and analysis hop sizes
		Rs = self.Rs
		Ra = int(self.Rs / float(alpha))

		# AM scaling factor due to window sliding
		wscale = sum([i**2 for i in w]) / float(Rs)
		L = x.size
		L0 = int(x.size*alpha)

		# Get an prior approximation of the fundamental frequency
		if alpha != 1.0:
			A = np.fft.fft(w*x[0:N])
			B = np.fft.fft(w*x[Ra:Ra+N])
			Freq0 = B/A * abs(B/A)
			Freq0[Freq0 == 0] = epsilon
		else:
			Freq0 = 1

		if alpha == 1.0: 	# we can fully retrieve the input (within numerical errors)
			# Place input signal directly over half of window
			x = np.append(np.zeros(N+Rs), x)
			x = np.append(x, np.zeros(N+Rs))

			# Initialize output signal
			y = np.zeros(x.size)
		else:
			x = np.append(np.zeros(Rs), x)
			#x = np.append(x, np.zeros(Rs))

			y = np.zeros(int((x.size)*alpha + x.size/Ra * alpha))

		# Pointers and initializations
		p, pp = 0, 0
		pend = x.size - (Rs+N)
		Yold = epsilon

		i = 0
		while p <= pend:
			i += 1
			# Spectra of two consecutive windows
			Xs = np.fft.fft(w*x[p:p+N])
			Xt = np.fft.fft(w*x[p+Rs:p+Rs+N])

			# Prohibit dividing by zero
			Xs[Xs == 0] = epsilon
			Xt[Xt == 0] = epsilon

			# inverse FFT and overlap-add
			if p > 0 :
				Y = Xt * (Yold / Xs) / abs(Yold / Xs)
			else:
				Y = Xt * Freq0

			Yold = Y
			Yold[Yold == 0] = epsilon

			y[pp:pp+N] += w*np.fft.ifft(Y)

			p = int(p+Ra)		# analysis hop
			pp += Rs			# synthesis hop

			sys.stdout.write ("Percentage finishied: %d %% \r" % int(100.0*p/pend))
			sys.stdout.flush()

		y = y / wscale


		if self.alpha == 1.0:
			# retrieve input signal perfectly
			x = np.delete(x, range(N+Rs))
			x = np.delete(x, range(x.size-(N+Rs), x.size))

			y = np.delete(y, range(N))
			y = np.delete(y, range(y.size-(N+2*Rs), y.size))
		else:
			# retrieve input signal perfectly
			x = np.delete(x, range(Rs))

			y = np.delete(y, range(Rs))
			y = np.delete(y, range(L0, y.size))

		return y

def sin_signal(self, fs, T, f0):
	"""
	Generate a sinusoidal signal
	fs: sampling frequency, T: signal duration, f0: signal frequency
	returns: sinusoid of frequency f0 and length T*fs
	"""
	t  = np.linspace(0, T, T*fs, endpoint=False)
	return 2**(16-2)*np.sin(2*np.pi * f0 * t)

def ramp_signal(self, fs, T):
	"""
	Generate a ramp signal
	fs: sampling frequency, T: signal duration, f0: signal frequency
	returns: ramp of length T*fs
	"""
	return 2**(16-2) * np.linspace(0, T, T*fs, endpoint=False) / (T*fs)

if __name__ == '__main__':

	if len(sys.argv) < 4:
		print "Usage: py.py <input_file.wav> <timestretch factor> <ouput_file.wav>"
	else:
		input_file = sys.argv[1]	# .wav input file
		alpha = float(sys.argv[2])	# timestrecth factor
		output_file = sys.argv[3]	# name of .wav output file

		# These parameters should be power of two for FFT
		N = 2**10					# Number of channels
		M = 2**10					# Size of window

		w = np.hanning(M-1)			# Type of Window (Hanning)
		#w = np.hamming(M-1)		# Type of Window (Hamming)
		#w = np.hamm(M-1)			# Type of Window (Hann)
		w = np.append(w, [0])		# Make window symmetric about (M-1)/2

		# Synthesis hop factor and hop size
		Os = 4.						# Synthesis hop factor
		Rs = int(N / Os)			# Synthesis hop size


		pv = PhaseVocoder(N, M, Rs, w, alpha)
		fs, x = pv.read_wav(input_file)	# wav input


		# Test with sinusoid or ramp input signal
		#x = ramp_signal(fs, 1.0)	# test signal
		#x = sin_signal(fs, 1.0, fs/float(N) * 4)

		# Timestretch by factor alpha
		y = pv.timestretch(x, alpha)

		# Pitchshift by a factor alpha
		# y = pv.pitchshift(x, alpha)

		# write to ouput file
		pv.write_wav(output_file, fs, y)

		# Uncomment this part if you wish to plot the input
		# and output signals
		import matplotlib.pyplot as plt
		# plot the input sound
		plt.subplot(2,1,1)
		plt.plot(np.arange(x.size)/float(fs), x)
		plt.axis([0, x.size/float(fs), min(x), max(x)])
		plt.ylabel('amplitude')
		plt.xlabel('time (sec)')
		plt.title('input sound: x')
		# plot the output sound
		plt.subplot(2,1,2)
		plt.plot(np.arange(y.size)/float(fs), y)
		plt.axis([0, y.size/float(fs), min(y), max(y)])
		plt.ylabel('amplitude')
		plt.xlabel('time (sec)')
		plt.title('output sound: x')
		plt.show()
