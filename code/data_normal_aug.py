import fnmatch  # Filtering filenames
import os

import numpy as np
import pandas as pd
from tqdm import tqdm as tqdm


def awgn(length, mean=0, std=1):
    eps = 10**-13
    noise = np.random.randn(length)
    noise = noise - noise.mean()
    noise /= (np.sqrt(noise.var() + eps))
    return noise * std + mean


def augment_data(path, n_aug=3):

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
                df[column] = df[column] + awgn(len(df[column]), mean=0, std=0.1)
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
    augment_data(path, n_aug=3)
