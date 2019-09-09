import fnmatch  # Filtering filenames
import os

import numpy as np
import pandas as pd
from tqdm import tqdm as tqdm

from utils.picklepickle import load_obj, save_obj
from utils.win_feats import compute_statistical_features as statfeat


def path_to_raw_data(path):
    # Get filenames
    filenames = []
    for root, dirnames, fnames in os.walk(path):
        for fname in fnmatch.filter(fnames, '*.csv'):
            filenames.append(os.path.join(root, fname))
    return filenames


def downsampling_signals(data, down_rate='25T'):

    data.index = pd.to_timedelta(data.index, unit='T')
    data = data.resample(down_rate).mean()

    return data


def proc_data(path=None,
              feat_path=None,
              force_feat_calc=False,
              filenames=None,
              down_rate='25T',
              window_size=4 * 10**3,
              column_labels=None):

    if column_labels is None:
        raise Exception(f"column labels!!!")
    if filenames is None:
        if path is None:
            raise Exception(f"you shoul've insert a valid path!")
        filenames = path_to_raw_data(path)

    if feat_path is None:
        if path is None:
            path = filenames[0]
        aux_path = os.path.dirname(path)
        feat_path = os.path.join(aux_path, '..', 'win_features')

    down_path = os.path.join(feat_path, 'downsampled_signal/', down_rate + '/')
    computed_feat_path = os.path.join(feat_path, 'computed_features/')

    for fn in tqdm(filenames):
        down_df_path = fn.replace(path, down_path).replace('.csv', '.pickle')
        if os.path.isfile(down_df_path):
            down_df = load_obj(down_df_path)
        else:
            raw_df = pd.read_csv(fn, header=None, names=column_labels)

            down_df = downsampling_signals(raw_df, down_rate=down_rate)

            dir_to_create = os.path.dirname(down_df_path)
            if (not os.path.exists(dir_to_create)):
                os.makedirs(dir_to_create)
            save_obj(obj=down_df, name=down_df_path)

        win_feat_df_path = fn.replace(path, computed_feat_path).replace('.csv', '.pickle')
        if os.path.isfile(win_feat_df_path) and (not force_feat_calc):
            continue
        else:
            win_feat_df = statfeat(data=down_df, window_size=window_size, center=False)

            dir_to_create = os.path.dirname(win_feat_df_path)
            if not os.path.exists(dir_to_create):
                os.makedirs(dir_to_create)
            save_obj(obj=win_feat_df, name=win_feat_df_path)


#*******************************************************************************
# Main
#*******************************************************************************

if __name__ == '__main__':

    current_dir = os.path.dirname(os.path.realpath(__file__))

    # Main parameters
    path = os.path.join(current_dir, "data/mafaulda/")
    feat_path = os.path.join(current_dir, "data/win_features")

    sign_nms = ['TCH', 'IAA', 'IRA', 'ITA', 'EAA', 'ERA', 'ETA', 'MIC']

    proc_data(path=path,
              feat_path=feat_path,
              window_size=4 * 10**3,
              column_labels=sign_nms,
              force_feat_calc=True)
