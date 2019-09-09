import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


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
    if len(y) == 0:
        return np.nan, np.nan

    # Setting output lists
    imax = []
    vmax = []

    # For each peak
    for i in range(npeaks):

        # Find maxima
        try:
            imax.append(np.nanargmax(y))
            vmax.append(np.nanmax(y))
        except:
            return np.nan, np.nan

        # Setting neighborhood as invalid
        y[imax[-1] - mpd:imax[-1] + mpd + 1] = np.nan

    # Finding sorting indices
    sidx = np.argsort(imax).tolist()

    # Sorting
    imax = [imax[i] for i in sidx]
    vmax = [vmax[i] for i in sidx]

    # Returning
    return imax, vmax


def freq_feat(data, which_feat, smp_frq=2e+3, frq_min=5, frq_max=700, num_frq=3, max_hrm=3):
    """
    Computes features from the raw data.

    @param data Input data list. Numpy matrices list.
    @param smp_frq Number of points. Scalar.
    @param frq_min Minimum target frequency. Scalar.
    @param frq_max Maximum target frequency. Scalar.
    @param num_frq Number of frequency on the interval.
    @param max_hrm Maximum harmonic on the interval.

    @return Preprocessed data.
    """

    if data.isnull().all().all == True:
        return np.nan
    else:
        # Finding the number of signals and samples
        signals = np.arange(data.shape[1])
        num_smp = data.shape[0]

        # Computing real DFT transform for each signal
        aux_data = [np.fft.rfft(data.iloc[:, sidx]) for sidx in signals]

        # Setting target frequency range
        frq_vls = np.linspace(0, smp_frq / 2.0, num=(num_smp / 2) + 1)

        frq_idx = np.logical_and(frq_vls >= frq_min, frq_vls <= frq_max)
        frq_vls = frq_vls[frq_idx]

        # Removing invalid frequencies
        aux_data = [np.real(np.abs(a[frq_idx])) for a in aux_data]

        pidx, _ = find_peaks(aux_data[0], mpd=15)
        if pidx is np.nan:
            return np.nan

        # Extract smallest frequency index and value (first harmonic)
        fidx = np.min(pidx)
        frot = frq_vls[fidx]

        # Setting the maximum harmonic
        max_fidx = min(max_hrm * fidx, aux_data[0].shape[0] - 4)

        # Harmonics index
        hfidx = np.int64(np.round(np.linspace(fidx, max_fidx, num_frq)))
        hfidx = hfidx.tolist()

        # Initializing the features vector
        frq_vec = []

        # For each signal
        for sidx, csgn in enumerate(aux_data):

            # Extracting harmonic components
            for ih, hf in enumerate(hfidx):

                # Extracting harmonic components
                rhf = range(hf - 3, hf + 4)
                frq_vec.append(np.max(csgn[rhf]))

        # Computing auxiliary values
        fmu = aux_data[0].mean()

        if which_feat == 'first_harmonic':
            return frq_vec[0]
        elif which_feat == 'second_harmonic':
            return frq_vec[1]
        elif which_feat == 'third_harmonic':
            return frq_vec[2]
        elif which_feat == 'freq_mean':
            return fmu
        elif which_feat == 'freq_rms':
            return np.sqrt(np.mean((aux_data[0])**2))
        elif which_feat == 'freq_std':
            return aux_data[0].std()
        elif which_feat == 'smallest_freq':
            return frot
        else:
            return -999
