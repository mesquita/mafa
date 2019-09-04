import os

import numpy as np
import pandas as pd

from prognosis.process.feature import find_peaks  # Peak detection


def freq_feat(data, smp_frq=5e+4, frq_min=5, frq_max=700, num_frq=3, max_hrm=3):
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

    # Finding the number of signals and samples
    signals = 1  #range(data.shape[1])
    num_smp = data.shape[0]

    # Computing real DFT transform for each signal
    aux_data = np.fft.rfft(data)

    # Setting target frequency range
    frq_vls = np.linspace(0, smp_frq / 2.0, num=(num_smp / 2) + 1)
    frq_idx = np.logical_and(frq_vls >= frq_min, frq_vls <= frq_max)
    frq_vls = frq_vls[frq_idx]

    # Removing invalid frequencies
    aux_data = [np.real(np.abs(a[frq_idx])) for a in aux_data]

    # Detect peaks from the tachometer
    pidx, _ = find_peaks(aux_data[0], mpd=15)

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
    fmu = [a.mean() for a in aux_data]

    # Computing the statistical features
    frq_vec.extend(fmu)
    frq_vec.extend([np.sqrt(np.mean(a**2)) for a in aux_data])
    frq_vec.extend([a.std() for a in aux_data])
    frq_vec.append(frot)
    frq_vec = np.r_[frq_vec].ravel()

    # Output
    return frq_vec


#*******************************************************************************
# Main
#*******************************************************************************

if __name__ == '__main__':

    current_dir = os.path.dirname(os.path.realpath(__file__))

    # Main parameters
    path = os.path.join(current_dir, "data/mafaulda/normal/12.288_0.csv")
    df = pd.read_csv(path, header=None)
    whole_signal = df[0]
    frq_vec = freq_feat(data=whole_signal)
    print('oi')
