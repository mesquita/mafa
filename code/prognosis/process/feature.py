#*******************************************************************************
# Universidade Federal do Rio de Janeiro
# Instituto Alberto Luiz Coimbra de Pos-Graduacao e Pesquisa de Engenharia
# Programa de Engenharia Eletrica
# Signal, Multimedia and Telecommunications Lab
#
# Author: Felipe Moreira Lopes Ribeiro, Luiz Gustavo Tavares
#
#*******************************************************************************

#*******************************************************************************
# feature.py
#
# Feature extraction/transformation methods.
#
# Created: 2015/10/09
# Modified: 2016/03/17
#
#*******************************************************************************
"""
feature.py

Feature extraction/transformation methods.
"""

#*******************************************************************************
# Imports
#*******************************************************************************

from fractions import Fraction  # Defines fractions

import numpy as np  # Numpy library (numeric algebra)
import scipy.signal as sgn  # Signal processing library
from numpy.lib.stride_tricks import as_strided  # For window function

#*******************************************************************************
# Functions
#*******************************************************************************


def roll_window(X, window_sz=512, step_sz=2):
    """
	Computes multidimensional rolling window.

	@param X Input signal [n_samples, n_features].
	@param window_sz Window size. Scalar.
	@param step_sz Step size. Scalar.

	@return New matrix
	"""

    # Extending matrix by window size
    if (len(X.shape) > 1):

        # Repeating first samples
        Xe = np.vstack((np.repeat(X[0:1, :], window_sz - 1, axis=0), X))

    else:

        # Repeating first samples
        Xe = np.concatenate((np.repeat(X[0], window_sz - 1, axis=0), X))

    # Computing number of windows
    num_win = np.ceil((Xe.shape[0] - window_sz + 1) / float(step_sz))

    # Testing number of dimensions
    if (len(Xe.shape) > 1):

        # Multidimensional matrix shape
        shape = (num_win, window_sz) + Xe.shape[-1:]
    else:

        # One dimensional shape
        shape = (num_win, window_sz)

    # Computing stride
    strides = (step_sz * Xe.strides[0], ) + Xe.strides

    # Returning new matrix
    return np.lib.stride_tricks.as_strided(Xe, shape=shape, strides=strides)


def mode_filter(X, window_sz=512, step_sz=1):
    """
	Computes multidimensional mode filter.

	@param X Input signal [n_samples, n_features]. Integer matrix.
	@param window_sz Window size. Scalar.
	@param step_sz Step size. Scalar.

	@return Filtered outputs.
	"""

    # Testing data length
    if (len(X.shape) > 1):

        # Recursive call along each axis
        Y = np.apply_along_axis(mode_filter, 0, X, window_sz, step_sz)

        # Concatenating resulting features
        Y = np.concatenate(Y, axis=1)

    else:

        # Computing window
        Xw = roll_window(X, window_sz, step_sz)

        # Finding minimum length
        min_len = X.max() - X.min()

        # Computing count
        Y = np.apply_along_axis(np.bincount, 1, Xw, minlength=min_len)
        Y = Y.argmax(axis=1)

    # Returning
    return Y


def time_feat(X, window_sz=512, step_sz=2):
    """
	Computes aggregated time features.

	@param X Input signal [n_samples, n_features].
	@param window_sz Window size. Scalar.
	@param step_sz Step size. Scalar.

	@return Aggregated time features.
	"""

    # Computing rolling window
    Xt = roll_window(X, window_sz, step_sz)

    # Reshaping new dimension
    Xt = Xt.reshape((Xt.shape[0], -1))
    return Xt


def roll_mean(X, window_sz=10, step_sz=1):
    """
	Computes multidimensional rolling mean.

	@param X Input signal [n_samples, n_features].
	@param window_sz Window size. Scalar.
	@param step_sz Step size. Scalar.

	@return Signal rolling mean.
	"""

    # Computing rolling window
    Xw = roll_window(X, window_sz, step_sz)

    # Computing mean over each window and returning
    Xm = np.mean(Xw, axis=1)
    return Xm


def roll_std(X, window_sz=10, step_sz=1):
    """
	Computes multidimensional rolling standard deviation.

	@param X Input signal [n_samples, n_features].
	@param window_sz Window size. Scalar.
	@param step_sz Step size. Scalar.

	@return Signal rolling standard deviation.
	"""

    # Computing rolling window
    Xw = roll_window(X, window_sz, step_sz)

    # Computing standard deviation over each window and returning
    Xs = np.std(Xw, axis=1)
    return Xs


def time_stat(X, window_sz=10, step_sz=1):
    """
	Computes multidimensional rolling statistics.

	@param X Input signal [n_samples, n_features].
	@param window_sz Window size. Scalar.
	@param step_sz Step size. Scalar.

	@return Signal rolling standard deviation.
	"""

    # Computing rolling window
    Xw = roll_window(X, window_sz, step_sz)

    # Computing statistics
    Xs = np.mean(Xw, axis=1)  # Rolling mean
    Xd = X - Xs
    Xs = np.concatenate((Xs, Xd), axis=1)  # Rolling difference
    Xs = np.concatenate((Xs, np.std(Xw, axis=1)), axis=1)  # Rolling std

    # Return
    return Xs


def fstft(X, window=np.hanning, window_sz=512, overlap=256):
    """
	Computes the Short-time Fourier transform for a given signal.

	@param X Input signal [nsamples].
	@param window Window function.
	@param window_sz Window size. Scalar.
	@param overlap Overlap size.

	@return Two elements:
		- Signal STFT;
		- FFT frequency range (unitary frequency).
	"""

    # Testing data length
    if (len(X.shape) > 1):

        # Recursive call along each axis
        Y, f = np.apply_along_axis(fstft, 0, X, window, window_sz, overlap)

        # Concatenating resulting features and saving the DFT frequencies
        Y = np.concatenate(Y, axis=1)
        f = f[0]
    else:
        # Computing function step size
        step_sz = max(window_sz - overlap, 1)

        # Computing windows function
        win = window(window_sz)

        # Computing rolling window and multiplying by the window function
        Xwin = np.copy(roll_window(X, window_sz, step_sz))
        Xwin *= win

        # Computing the DFT coefficients power and frequencies
        Y = np.real(np.absolute(np.fft.rfft(Xwin, n=window_sz, axis=1)))
        f = np.fft.rfftfreq(window_sz)

    # Returning
    return Y, f


def stft(X, window=np.hanning, ovlFactor=0.75, fftSize=40980):
    """
	Computes the Shot-Time Fourier Transform (STFT) of a signal
	@param X Input signal [1-D array].
	@param window Sampling window (default=hanning).
	@param ovlFactor Sampling window overlap percentual (default=0.75).
	@param fftSize Sampling window size (default=64).

	@return Output spectrum.
	"""

    # Indice difference for each window
    step_size = np.int32(np.floor(fftSize * (1 - ovlFactor)))

    # Number of windows
    numWin = np.int32(np.ceil(len(X) / np.float32(step_size)))

    # Window function
    #window = np.resize(np.hanning(fftSize),(fftSize,1))
    window = np.hanning(fftSize)

    # Zero padding
    #smp = np.append(np.zeros(np.floor(fftSize/2.0)), X)
    #smp = np.append(smp, np.zeros(fftSize))
    smp = X

    # Computing frames
    xstr = (smp.strides[0] * step_size, smp.strides[0])

    frames = as_strided(smp, shape=(numWin, fftSize),\
     strides= xstr).copy()

    # Multiplying by window
    frames *= window

    # Computing DFT and returning
    Y = np.real(np.absolute(np.fft.rfft(frames)))

    return Y

def stft2(X, window=np.hanning, ovlFactor=0.75, winSize=40980,\
 fftSize=40980):
    """
	Computes the Shot-Time Fourier Transform (STFT) of a signal
	@param X Input signal [1-D array].
	@param window Sampling window (default=hanning).
	@param ovlFactor Sampling window overlap percentual (default=0.75).
	@param winSize FFT window size.
	@param fftSize Sampling window size (default=64).

	@return Output spectrum.
	"""

    # Testing dimension
    if X.ndim == 1:

        # Add new axis
        X = X[:, np.newaxis]

    # Indice difference for each window
    step_size = np.int32(np.floor(winSize * (1 - ovlFactor)))

    # Window
    win = window(winSize)

    # Number of windows
    numWin = np.int32(np.ceil(len(X) / np.float32(step_size)))

    # Output spectrum
    Y = np.empty([0, fftSize / 2 + 1])

    # Zero Padding
    X = np.vstack((np.zeros((winSize / 2, 1)), X))
    X = np.vstack((X, np.zeros((winSize / 2, 1))))

    # Computing windows
    for k in range(numWin):
        X_row = X[(k * step_size):(k * step_size + winSize), 0] * win
        Y = np.vstack((np.fft.rfft(X_row, fftSize), Y))

    # Converting to absolute and real values
    Y = np.real(np.abs(Y))

    return Y


def norm_spec(Y, norm_frq, rot_frq=None, smp_frq=250000):
    """
	Normalizes a frequency signal using auxiliary information.

	@param Y Input data.
	@param norm_frq Resulting rotation Frequency.
	@param rot_frq Rotation frequency signal.
	@param smp_frq Signal sampling frequency.

	@return Normalized signal.
	"""

    # Extracting data size and initializing ouput matrix
    Y_norm = np.zeros_like(Y)

    # Getting matrix shape
    nrows, ncols = np.shape(Y)

    # Frequency vector
    freq = np.linspace(0, smp_frq / 2, num=ncols, endpoint=True)

    # Testing rotation frequency signal
    if rot_frq is None:

        # Computing rotation frequency indices
        peak_idx = np.argmax(np.absolute(Y), axis=1)
        peak_val = freq[peak_idx]

    else:
        peak_val = rot_frq

    # For each frame
    for i in range(nrows):

        # Extracting current frame in time domain
        time_row = np.fft.irfft(Y[i, :])

        # Finding factor
        ratio = peak_val / norm_frq
        #print 'ratio = ',ratio
        #frac = Fraction.from_float(ratio).limit_denominator(1000)

        # Extracting numerator and denominator
        #p = frac.numerator
        #q = frac.denominator

        # Resampling
        new_row = sgn.resample(time_row, len(time_row) * ratio)

        # Normalizing
        Y_norm[i, :] = np.real(np.absolute(np.fft.rfft(new_row,\
         2*len(Y[i, :])-1)))

    # Returning
    return Y_norm


def find_peaks(x, npeaks=4, mpd=3):
    """
	Finds npeaks greatest local maximas.

	@param x Input sequence.
	@param npeaks Number of valid peaks.
	@param mpd Minimum peak distance (in samples).

	@return Two elements:
		- List with valid peaks indices; and
		- List with respective maximum values.
	"""

    # Copy input
    y = np.copy(x)

    # Setting output lists
    imax = []
    vmax = []

    # For each peak
    for i in range(npeaks):

        # Find maxima
        imax.append(np.nanargmax(y))
        vmax.append(np.nanmax(y))

        # Setting neighborhood as invalid
        y[imax[-1] - mpd:imax[-1] + mpd + 1] = np.nan

    # Finding sorting indices
    sidx = np.argsort(imax).tolist()

    # Sorting
    imax = [imax[i] for i in sidx]
    vmax = [vmax[i] for i in sidx]

    # Returning
