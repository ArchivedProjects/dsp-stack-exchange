import math
import asyncio
import numpy as np

from rtlsdr import RtlSdr
from scipy.io import wavfile
from datetime import datetime

def demodulate_fm_mono(x, df=1.0, fc=0.0):
	''' Perform FM demodulation of complex carrier.
		Stolen From: https://stackoverflow.com/a/60208259/6828099

	Args:
		x (array):  FM modulated complex carrier.
		df (float): Normalized frequency deviation [Hz/V].
		fc (float): Normalized carrier frequency.

	Returns:
		Array of real modulating signal.
	'''

	# Remove carrier.
	# https://numpy.org/doc/stable/reference/generated/numpy.arange.html
	# https://pythonnumericalmethods.berkeley.edu/notebooks/chapter24.03-Fast-Fourier-Transform.html
	n = np.arange(len(x))  # Generate an Evenly Distributed Range Of Numbers From 0 To Number Of Samples
	rx = x*np.exp(-1j*2*np.pi*fc*n)  # This is part of the Fast Fourier Transform Equation

	# Extract phase of carrier.
	# https://numpy.org/doc/stable/reference/generated/numpy.arctan2.html
	phi = np.arctan2(np.imag(rx), np.real(rx))  # Arc Tangent Of Two Arrays ???

	# Calculate frequency from phase.
	# https://numpy.org/doc/stable/reference/generated/numpy.diff.html
	# https://numpy.org/doc/stable/reference/generated/numpy.unwrap.html
	y = np.diff(np.unwrap(phi)/(2*np.pi*df))  # ???

	return y

# TODO: Learn What These Equations Mean: https://ocw.mit.edu/courses/electrical-engineering-and-computer-science/6-661-receivers-antennas-and-signals-spring-2003/lecture-notes/lecture17.pdf
def demodulate_am_mono(x, fc=0.0):
	# Acm(t)cos(ωct)⋅2cos(ωct)
	# Received = y(t) = A[1+m*s(t)]*c(t)+n(t)*c(t)+m(t)*z(t)
	# s(t) = sin(w) t <= 1 where w sub m
	# c(t) = cos(w) t where w sub c
	# n(t) = n(t) where n sub c
	# m(t) = n(t) where n sub s
	# z(t) = sin(w) t where w sub c
	
	# 100 / 0.1 = 1000
	# 25166 * 0.1 = stop
	# start, stop, step
	
	# No Idea What's Going On With This: https://dsp.stackexchange.com/q/48026/25247
	
	message_freq = sdr.center_freq  # 50
	
	n = np.arange(len(x))
	rx = x*np.exp(-1j*2*np.pi*fc*n)
	
	# Remove carrier.
	#n = np.arange(len(x))
	#rx = x*np.exp(-1j*2*np.pi*fc*n)

	# Extract phase of carrier.
	#phi = np.arctan2(np.imag(rx), np.real(rx))

	# Calculate frequency from phase.
	#return np.diff(np.unwrap(phi)/(2*np.pi*df))
	
	return wav_samples

# TODO: Figure out how to stream to file instead of this mess!!!
def write_no_demod(data, no_demod, rate: int):
	wav_samples = np.zeros((len(data), 2), dtype=np.float32)
	wav_samples[...,0] = data.real
	wav_samples[...,1] = data.imag
	
	if no_demod is not None:
		no_demod = np.concatenate((no_demod, wav_samples))
	else:
		no_demod = wav_samples

	wavfile.write('no_demod.wav', int(rate), no_demod)
	return no_demod

def write_fm_demod(data, fm_demod, rate: int):
	if fm_demod is not None:
		fm_demod = np.concatenate((fm_demod, data))
	else:
		fm_demod = data

	wavfile.write('fm.wav', int(rate), demodulate_fm_mono(x=fm_demod))
	return fm_demod

def write_am_demod(data, am_demod, rate: int):
	if am_demod is not None:
		am_demod = np.concatenate((am_demod, data))
	else:
		am_demod = data

	wavfile.write('am.wav', int(rate), demodulate_am_mono(x=am_demod))
	return am_demod

async def save_audio(sdr, rate=48e3):
	import scipy.signal as sps

	no_demod = None
	demod_one = None
	start_time = datetime.now()
	print("Start Recording: %s" % start_time)
	async for samples in sdr.stream():
		current_time = datetime.now()
		print("Current Time Passed: %s" % (current_time - start_time), end="\r")
		try:
			number_of_samples = round(len(samples) * float(rate) / sdr.sample_rate)
			data = sps.resample(samples, number_of_samples)
			
			no_demod = write_no_demod(data, no_demod, rate=rate)
			#demod_one = write_fm_demod(data, demod_one, rate=rate)
			demod_one = write_am_demod(data, demod_one, rate=rate)
		except KeyboardInterrupt:
			break
			
	stop_time = datetime.now()
	print()
	print("Stop Recording: %s" % stop_time)
	await sdr.stop()
	
	sdr.close()
	
	print("Record Time: %s" % (stop_time - start_time))

if __name__ == "__main__":
	device = 0;

	devices = RtlSdr.get_device_serial_addresses()
	if len(devices) == 0: raise Exception("No Detected RTLSDR Devices!!!");

	# Find the device index for a given serial number
	device_index = RtlSdr.get_device_index_by_serial(devices[device]) # You can insert your Serial Address (as a string) directly here
	sdr = RtlSdr(device_index)
	
	sdr.sample_rate = 1e6  # 1,000,000 Hz (How Many Samples Per Second)
	#sdr.center_freq = 315e6  # 315,000,000 Hz
	sdr.center_freq = 314873e3  # 314,873.000 kHz
	#sdr.center_freq = 1079e5
	sdr.gain = 'auto'
	
	loop = asyncio.get_event_loop()
	loop.run_until_complete(save_audio(sdr=sdr, rate=48e3))
	loop.close()
