import fnmatch  # Filtering filenames
import os

import numpy as np
import pandas as pd
from tqdm import tqdm as tqdm


def add_noise_snr(signal, target_snr_db=20):
    """Add noise level accordingly to target_snr_db.

    Args:
        signal (numpy.array): signal whose snr level will be changed.
        target_snr_db (int, optional): disered snr level. Defaults to 20.

    Returns:
        numpy.array: noisy signal with SNR = target_snr_db (in dB).
    """
    x_volts = signal
    x_watts = x_volts**2

    # Calculate signal power and convert to dB
    sig_avg_watts = np.mean(x_watts)
    sig_avg_db = 10 * np.log10(sig_avg_watts)
    # Calculate noise according to [2] then convert to watts
    noise_avg_db = sig_avg_db - target_snr_db
    noise_avg_watts = 10**(noise_avg_db / 10)
    # Generate an sample of white noise
    mean_noise = 0
    noise_volts = np.random.normal(mean_noise, np.sqrt(noise_avg_watts), len(signal))
    # Noise up the original signal
    y_volts = x_volts + noise_volts
    return y_volts


def augment_data(path, n_aug=3):
    """Adds noise do normal signals.

    Args:
        path (string): path to data files.
        n_aug (int, optional): how many different files are going to be
        created from the original one. Defaults to 3.
    """
    # Get filenames
    filenames = []
    filenames_out = []
    for root, dirnames, fnames in os.walk(path):
        root_aux = root + '_aug'
        for fname in fnmatch.filter(fnames, '*.csv'):
            filenames.append(os.path.join(root, fname))
            filenames_out.append(os.path.join(root_aux, fname))

    # Parsing filenames
    i = 1
    fim = 49 * n_aug
    for fn, fn_out in zip(filenames, filenames_out):
        df = pd.read_csv(fn, header=None)
        for ii in range(n_aug):
            for column in df.columns:
                df[column] = add_noise_snr(signal=df[column], target_snr_db=20)
            path_out = fn_out[:-4] + '_' + str(ii) + '.csv'
            df.to_csv(path_out, header=False, index=False)
            print(f'{i} de {fim}')
            i += 1


#*******************************************************************************
# Main
#*******************************************************************************

if __name__ == '__main__':

    current_dir = os.path.dirname(os.path.realpath(__file__))

    # Main parameters
    path = os.path.join(current_dir, "data/mafaulda/normal")
    augment_data(path, n_aug=8)
