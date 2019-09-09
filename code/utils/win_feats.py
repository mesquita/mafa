import pandas as pd
from scipy import stats

from utils.freq_feats import freq_feat


def compute_statistical_features(data, window_size, center=False):
    '''Extração de características statísticas em uma janela deslizante de tamanho
    'window_size'.

    Arguments:
        data {pandas.dataframe} -- [Pandas dataframe contendo onde serão computadas as estatísticas]
        window_size {int} -- [Tamanho da janela da qual serão computadas as features]

    Keyword Arguments:
        center {bool} -- [description] (default: {False})

    Returns:
        [pandas.dataframe] -- [dataframe contendo as features.]
    '''
    window = data.rolling(window_size, center=center)

    # compute features
    mean_feats = window.mean().add_suffix('_mean')
    median_feats = window.median().add_suffix('_median')
    std_feats = window.std().add_suffix('_std')
    min_feats = window.min().add_suffix('_min')
    max_feats = window.max().add_suffix('_max')
    skew_feats = window.apply(stats.skew, raw=False).add_suffix('_skew')
    kurt_feats = window.apply(stats.kurtosis, raw=False).add_suffix('_kurtosis')
    first_percentile_feats = window.quantile(0.25).add_suffix('_1st-Percentile')
    third_percentile_feats = window.quantile(0.75).add_suffix('_3rd-Percentile')

    first_harmonic_func = lambda x: freq_feat(data=x.to_frame(), which_feat='first_harmonic')
    first_harmonic = window.apply(first_harmonic_func, raw=False).add_suffix('_freq-1st-harm')

    second_harmonic_func = lambda x: freq_feat(data=x.to_frame(), which_feat='second_harmonic')
    second_harmonic = window.apply(second_harmonic_func, raw=False).add_suffix('_freq-2st-harm')

    third_harmonic_func = lambda x: freq_feat(data=x.to_frame(), which_feat='third_harmonic')
    third_harmonic = window.apply(third_harmonic_func, raw=False).add_suffix('_freq-3st-harm')

    freq_mean_func = lambda x: freq_feat(data=x.to_frame(), which_feat='freq_mean')
    freq_mean = window.apply(freq_mean_func, raw=False).add_suffix('_freq-mean')

    freq_rms_func = lambda x: freq_feat(data=x.to_frame(), which_feat='freq_rms')
    freq_rms = window.apply(freq_rms_func, raw=False).add_suffix('_freq-rms')

    freq_std_func = lambda x: freq_feat(data=x.to_frame(), which_feat='freq_std')
    freq_std = window.apply(freq_std_func, raw=False).add_suffix('_freq-std')

    # concatenate all of them
    feats = pd.concat(
        [
            mean_feats, median_feats, std_feats, min_feats, max_feats, skew_feats, kurt_feats,
            first_percentile_feats, third_percentile_feats, first_harmonic, second_harmonic,
            third_harmonic, freq_mean, freq_rms, freq_std
        ],
        axis=1,
    )

    # drop all NaN rows
    # feats.dropna(axis=0, how='all', inplace=True)
    return feats
