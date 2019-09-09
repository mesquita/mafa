import os

import pandas as pd


def unifying_features(path_to_save, wavelet_feat_path, stats_feat_path):

    wavelet_feat = pd.read_csv(wavelet_feat_path)
    stats_feat = pd.read_csv(stats_feat_path)

    all_feat = pd.merge(wavelet_feat, stats_feat, on='Sample')

    all_feat.to_csv(path_to_save, index_label='Sample')


#*******************************************************************************
# Main
#*******************************************************************************
if __name__ == '__main__':

    current_dir = os.path.dirname(os.path.realpath(__file__))
    wavelet_feat_path = os.path.join(current_dir, "data/data_wavelet.csv")
    stats_feat_path = os.path.join(current_dir, "data/data.csv")
    path_to_save = os.path.join(current_dir, "data/data_all_feat.csv")
    unifying_features(path_to_save, wavelet_feat_path, stats_feat_path)
