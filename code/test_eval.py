import os
import pickle
from collections import Counter

import numpy as np
import pandas as pd
from bayes_opt import BayesianOptimization
from bayes_opt.util import Colours
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import (GridSearchCV, cross_val_score, train_test_split)

from utils.confusionmatrix import plot_confusion_matrix


def open_data(path):
    """Open the csv data.

    Args:
        path (string): string that is the path to csv.

    Returns:
        pandas.DataFrame: dataframe with data from csv.
    """
    data = pd.read_csv(path)
    return data


def load_obj(name):
    with open(name, 'rb') as f:
        return pickle.load(f)


def evaluation(X_train, y_train, X_val, y_val, best_param_path=None):
    """Evaluation of training using training data and validation data.

    Args:
        X_train (pandas.DataFrame): dataframe containing training data.
        y_train (pandas.Series): series containing training labels.
        X_val (pandas.DataFrame): dataframe containing validation data.
        y_val (pandas.Series): series containing validation labels.
        path_to_best_param (string, optional): Path to where the pickle
        with the best parameters are. Defaults to None.
    """

    if best_param_path is None:
        best_param_path = "best_params.pickle"

    best_params = load_obj(best_param_path)

    clf_best = RandomForestClassifier(**best_params)
    clf_best.fit(X_train, y_train)

    y_pred = clf_best.predict(X_val)

    print(classification_report(y_val, y_pred, target_names=labels))
    plot_confusion_matrix(data_true=y_val,
                          data_pred=y_pred,
                          classes=labels,
                          title='Confusion Matrix Test',
                          normalize=True,
                          save_plot=True)


#*******************************************************************************
# Main
#*******************************************************************************

if __name__ == '__main__':
    # Labels

    labels = ['normal', 'imbalance', 'horizontal-misalignment',\
    'vertical-misalignment', 'underhang', 'overhang']

    dataset = ['X_train', 'y_train', 'X_val', 'y_val', 'X_test', 'y_test']

    current_dir = os.path.dirname(os.path.realpath(__file__))
    data_dict = dict.fromkeys(dataset, 0)
    for datatype in dataset:
        whole_path = os.path.join(current_dir, 'data', datatype + '.csv')
        df = open_data(whole_path)
        data_dict[datatype] = df

    evaluation(data_dict['X_train'],
               data_dict['y_train'],
               data_dict['X_test'],
               data_dict['y_test'],
               best_param_path=None)
