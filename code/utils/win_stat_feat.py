import pandas as pd
from scipy import stats


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

    # concatenate all of them
    feats = pd.concat(
        [
            mean_feats, median_feats, std_feats, min_feats, max_feats, skew_feats, kurt_feats,
            first_percentile_feats, third_percentile_feats
        ],
        axis=1,
    )

    # drop all NaN rows
    # feats.dropna(axis=0, how='all', inplace=True)
    return feats
