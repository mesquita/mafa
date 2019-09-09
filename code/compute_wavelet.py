import fnmatch  # Filtering filenames
import os

import numpy as np  # Numpy library (numeric algebra)
import pandas as pd  # Pandas library (csv reader, etc...)
import pywt
from tqdm import tqdm as tqdm

from utils.picklepickle import load_obj, save_obj
from utils.wavelets_utils import (cols_namer, fill_data, swt_decomposition,
                                  truncate_signal)


def save_col_names(col_names, level, wavelet_name):
    current_dir = os.path.dirname(os.path.realpath(__file__))
    path = os.path.join(current_dir, "data/col_names/",
                        str(level) + '_' + wavelet_name + '_'
                        "cols.pickle")

    save_obj(obj=col_names, name=path)


def wvl_feat(data, level=4, wavelet_name='bior6.8'):
    wavelet = pywt.Wavelet(wavelet_name)
    df_data, cutting_size = truncate_signal(data, level)
    decomposition = swt_decomposition(df_data, wavelet, level)
    cols = df_data.columns
    cols_name = cols_namer(cols, wavelet_name, level)

    # save_col_names(cols_name, level, wavelet_name)

    feats_waves = pd.DataFrame(decomposition.T, columns=cols_name)

    return feats_waves


def read_mafaulda(path, wvl_level=4, wvl_name='bior6.8'):
    # Get filenames
    sign_nms = ['TCH', 'IAA', 'IRA', 'ITA', 'EAA', 'ERA', 'ETA', 'MIC']
    filenames = []
    wavelet_feat_path = path.replace('mafaulda', 'mafaulda_wavelet')
    for root, dirnames, fnames in os.walk(path):
        for fname in fnmatch.filter(fnames, '*.csv'):
            filenames.append(os.path.join(root, fname))

    for fn in tqdm(filenames):
        #  Reading data and converting to float
        data = pd.read_csv(fn, header=None, names=sign_nms)
        wvl_df = wvl_feat(data, level=wvl_level, wavelet_name=wvl_name)
        path_to_save = fn.replace(path, wavelet_feat_path).replace('.csv', '.pickle')
        save_obj(obj=wvl_df, name=path_to_save)


#*******************************************************************************
# Main
#*******************************************************************************

if __name__ == '__main__':

    current_dir = os.path.dirname(os.path.realpath(__file__))

    # Main parameters
    path = os.path.join(current_dir, "data/mafaulda/")

    # Reading data and saving
    read_mafaulda(path)
