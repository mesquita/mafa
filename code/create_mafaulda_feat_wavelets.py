#***********************************************************************************
# Imports
#***********************************************************************************

import fnmatch  # Filtering filenames
import glob  # Unix style pathname pattern expansion
import os  # Parsing and walking through the data directory

# Basic imports
import numpy as np  # Numpy library (numeric algebra)
import pandas as pd  # Pandas library (csv reader, etc...)
# Statistics
from scipy.stats import entropy, kurtosis, skew  # Kurtosis and Entropy
from sklearn.neighbors import KernelDensity  # Kernel density estimator
from tqdm import tqdm as tqdm

from prognosis.process import feature as feat  # Import feature methods
from utils.picklepickle import load_obj


def data_entropy(X, n_grid=1000, kernel=False, bandwidth=0.2, **kwargs):
    """
    Computes unidimensional entropy from data points.

    @param X Input matrix [n_samples, n_features].
    @param n_grid Number of grid points. Integer.
    @param kernel Boolean to set kernel method on/off.
    @param bandwidth Kernel bandwidth. Scalar.

    @return Vector of [n_features], with the corresponding feature entropy.
    """

    # Testing X dimension
    if (len(X.shape) > 1):

        # Computing per axis
        ent_h = np.apply_along_axis(data_entropy, 0, X, n_grid, kernel,\
         bandwidth, **kwargs)
        ent_h = np.array(ent_h)

    else:

        # Finding sample range
        x_max = X.max()
        x_min = X.min()

        # Testing for kernel method
        if kernel:

            # Computing random sampling
            rnd_idx = np.random.choice(X.shape[0], size=n_grid,\
             replace=False)

            # Kernel density estimation
            kde = KernelDensity(bandwidth=bandwidth, **kwargs)
            kde.fit(X[rnd_idx, np.newaxis])

            # Computing distro
            x_grid = np.linspace(x_min, x_max, n_grid)
            pdf = kde.score_samples(x_grid[:, np.newaxis])  # Log-likelihood
            pdf = np.exp(pdf)  # Distribution estimation

        else:

            # Computing grid
            x_grid = np.arange(x_min, x_max, bandwidth)

            # Computing histogram
            pdf, _ = np.histogram(X, bins=x_grid, density=True)

        # Computing entropy
        ent_h = entropy(pdf)

    # Return entropy
    return ent_h


def stat_feat(data):
    """
    Computes the statistical feats from the raw data. Based on the features
    found in [Rauber2015].

    @param data Input data. Numpy matrix.

    @return Statistical features.
    """

    # Computing auxiliary values
    rms = np.sqrt(np.mean(data**2, axis=0))  # RMS value
    adt = np.abs(data)  # Data absolute value
    sra = np.mean(np.sqrt(adt), axis=0)**2  # Square root of the amplitude
    krt = kurtosis(data, axis=0)

    # Computing the statistical features
    sts_vec = []
    sts_vec.append(data_entropy(data))  # Entropy
    sts_vec.append(rms)  # RMS value
    sts_vec.append(sra)  #SRA
    sts_vec.append(krt)  # Kurtosis
    sts_vec.append(skew(data, axis=0))  # Skew
    sts_vec.append(data.max(axis=0) - data.min(axis=0))  # Peak2peak value
    sts_vec.append(adt.max(axis=0) / rms)  # Crest factor
    sts_vec.append(np.max(adt, axis=0) / np.mean(adt, axis=0))  # Impulse factor
    sts_vec.append(adt.max(axis=0) / sra)  # Margin factor
    sts_vec.append(rms / np.mean(adt, axis=0))  # Shape factor
    sts_vec.append(krt / rms)  # Kurtosis factor
    sts_vec.append(data.mean(axis=0))  # Mean value
    sts_vec.append(data.std(axis=0))  # Standard Deviation

    # Converting statistical features to numpy array
    sts_vec = np.r_[sts_vec].ravel()

    # Returning
    return sts_vec


def read_mafaulda(path, column_names):

    stat_nms = ["Ent_", "RMS_", "SRA_", "Krt_", "Skw_", "PPK_", "CF_", "ImF_",\
    "MrF_", "ShF_", "KrF_", "Mu_", "SD_"]

    # Status types
    sts_type = ['normal', 'imbalance', 'horizontal-misalignment',\
    'vertical-misalignment', 'underhang', 'overhang']
    head_nms = []
    head_nms.extend([p + s for p in stat_nms for s in column_names])
    head_nms.extend(["Class"])
    # Get filenames
    filenames = []
    for root, dirnames, fnames in os.walk(path):
        for fname in fnmatch.filter(fnames, '*.pickle'):
            filenames.append(os.path.join(root, fname))

    out_data = []
    for fn in tqdm(filenames):

        # Parsing filename
        aux_fn = os.path.normpath(fn)
        aux_fn = aux_fn.split(os.sep)

        # Finding class
        cur_clss = aux_fn[int(np.argwhere([s in sts_type for s in aux_fn]))]
        cur_clss = sts_type.index(cur_clss)

        cur_raw = load_obj(fn)
        cur_raw = cur_raw.values.astype(float)
        sts_vec = stat_feat(cur_raw)
        output = np.r_[sts_vec, cur_clss]
        out_data.append(output.copy())

    out_data = pd.DataFrame(out_data, columns=head_nms)

    return out_data


def open_col_names(level, wavelet_name):
    current_dir = os.path.dirname(os.path.realpath(__file__))
    path = os.path.join(current_dir, "data/col_names/",
                        str(level) + '_' + wavelet_name + '_'
                        "cols.pickle")

    return load_obj(path)


#*******************************************************************************
# Main
#*******************************************************************************

if __name__ == '__main__':

    current_dir = os.path.dirname(os.path.realpath(__file__))

    # Main parameters
    path = os.path.join(current_dir, "data/mafaulda_wavelet/")

    feat_path = os.path.join(current_dir, "data/data_wavelet.csv")  # Output feature path

    # Reading data and saving
    col_names = open_col_names(level=4, wavelet_name='bior6.8')
    data = read_mafaulda(path, col_names)
    data.to_csv(feat_path, index_label='Sample')
