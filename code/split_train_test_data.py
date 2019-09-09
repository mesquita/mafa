import os
import pickle
import random
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from utils.picklepickle import load_obj, save_obj


def get_data_and_label(data, type_of_data):
    """Separate the features from labels

    Args:
        data (pandas.DataFrame): dataframe containing both features and labels.
        Labels must be in a column called 'Class', that is the default.
        type_of_data (str): string saying the type_of_data from where the data came.

    Returns:
        X, y: dataframe with only features and series with only labels.
    """
    if type_of_data == 'wavelet':
        X = data.drop(labels=['Class'], axis=1)
    else:
        X = data.drop(labels=['Fr', 'RotFreq', 'Weigth', 'Shift', 'Bearing', 'Class'], axis=1)

    y = data['Class']

    return X, y


def compute_label_percentage(y):
    """Compute the label percentage containing in y

    Args:
        y (pandas.Series): Series containing labels.

    Returns:
        list: list that has percentage of labels
    """
    # get the count of how many failures of each type y has and turn it into
    # a numpy array
    label_count_array = np.fromiter(Counter(y).values(), dtype=float)

    # sum the total
    total = np.sum(label_count_array)

    # return the percentage
    return (label_count_array / total) * 100


def make_label_distr_df(y, y_train, y_test, labels):
    """Make pandas.Dataframe with label percentage in each input.

    Args:
        y (pandas.Series/numpy.array): series/array containing all the labels
        y_train (pandas.Series/numpy.array): series/array containing training labels
        y_test (pandas.Series/numpy.array): series/array containing test labels
        labels (list): list containing labels names

    Returns:
        pandas.DataFrame: dataframe containing ordered label percentage in each case.
    """

    # create empty dataframe
    df = pd.DataFrame(columns=['y', 'y_train', 'y_test'], index=labels)

    # populate with percentage
    df['y'] = compute_label_percentage(y)
    df['y_train'] = compute_label_percentage(y_train)
    df['y_test'] = compute_label_percentage(y_test)

    return df


def sep_train_val_test(df, type_of_data, labels):
    """Separtes train, validation and test sets.

    Args:
        df (pandas.DataFrame): dataframe containig all the data and labels.
        labels (list): list containing each label name.

    Returns:
        X_train, y_train, X_test, y_test: dataframes and series containing data and labels.
    """

    # separates features and label
    X, y = get_data_and_label(df, type_of_data)

    # separate train & test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=33)

    df = make_label_distr_df(y=y, y_train=y_train, y_test=y_test, labels=labels)

    # save the X_ and y_ (train, test)
    dataset = ['X_train', 'y_train', 'X_test', 'y_test']
    for datatype in dataset:
        whole_path = os.path.join(current_dir, 'data', 'data_sep', type_of_data,
                                  datatype + '.pickle')
        save_obj(obj=(vars()[datatype]), name=whole_path)

    print('Porcentagem das separações:')
    print(df)
    df.to_csv(path_or_buf=os.path.join(current_dir, "data/sep_percen_" + type_of_data + "_.csv"))


#*******************************************************************************
# Main
#*******************************************************************************
if __name__ == '__main__':

    random.seed(33)

    # Labels
    labels = ['normal', 'imbalance', 'horizontal-misalignment',\
    'vertical-misalignment', 'underhang', 'overhang']

    # Data path parameters
    current_dir = os.path.dirname(os.path.realpath(__file__))

    type_of_data = 'wavelet'  # 'wavelet' / 'raw' / 'all_feat'
    feat_path = os.path.join(current_dir, "data/data_" + type_of_data + ".csv")

    # Open features and label data
    df = pd.read_csv(feat_path, index_col='Sample')

    # Separate train, validation and test
    sep_train_val_test(df=df, type_of_data=type_of_data, labels=labels)
