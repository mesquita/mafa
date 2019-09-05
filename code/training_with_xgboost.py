import fnmatch  # Filtering filenames
import os
import pickle
from collections import Counter

import numpy as np
import pandas as pd
import xgboost as xgb
from bayes_opt import BayesianOptimization
from bayes_opt.util import Colours
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from utils.confusionmatrix import plot_confusion_matrix
from utils.picklepickle import load_obj, save_obj


def open_train_val_data(path):
    filenames = []
    sep_data = {}
    for root, dirnames, fnames in os.walk(path):
        for num, fname in enumerate(fnmatch.filter(fnames, '*.pickle')):
            filenames.append(os.path.join(root, fname))
            name = fname[:-7]
            sep_data[name] = load_obj(filenames[num])

    X_train = sep_data['X_train']
    y_train = sep_data['y_train']
    X_test = sep_data['X_test']
    y_test = sep_data['y_test']
    return X_train, y_train, X_test, y_test


def optimize_xgb(X_train, y_train, param_dict, init_points=3, n_iter=5, nfold=3):
    dtrain = xgb.DMatrix(X_train, label=y_train)

    def xgb_crossval(max_depth, eta, gamma, colsample_bytree, num_boost_round):
        cv_result = xgb.cv(
            {
                'eval_metric': 'rmse',
                'max_depth': int(max_depth),
                'subsample': 0.8,
                'eta': eta,
                'gamma': gamma,
                'colsample_bytree': colsample_bytree
            },
            dtrain,
            num_boost_round=int(num_boost_round),
            nfold=nfold)

        return -1.0 * cv_result['test-rmse-mean'].iloc[-1]

    xgb_bo = BayesianOptimization(xgb_crossval, pbounds=param_dict, verbose=2)

    xgb_bo.maximize(init_points=init_points, n_iter=n_iter, acq='ei')

    best_params = xgb_bo.max['params']

    return best_params


def train(X_train,
          y_train,
          param_to_search,
          init_points=3,
          n_iter=5,
          kfold=10,
          best_param_path=None):
    """Training a Random Forest Classifier. Currently saving the best parameters
    to a pickle file named "best_params.pickle".

    Args:
        X_train (pandas.DataFrame): dataframe containing training data.

        y_train (pandas.Series): series containing training labels.
        gridsearch (bool, optional): [description]. Defaults to False.

        param_to_search (dict, optional): Dictionary containing parameters
        that you intended the search to lookt at. Defaults to None.

        kfold (int, optional): Integer number that says the number of folds, it
        is the k in kfold. Defaults to None.
    """

    if best_param_path is None:
        best_param_path = "best_params_xgb.pickle"

    best_params = optimize_xgb(X_train=X_train,
                               y_train=y_train,
                               param_dict=param_to_search,
                               init_points=init_points,
                               n_iter=n_iter,
                               nfold=kfold)
    save_obj(obj=best_params, name=best_param_path)


def evaluation(X_train, y_train, X_test, y_test, labels, best_param_path=None):
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
        best_param_path = "best_params_xgb.pickle"

    best_params = load_obj(best_param_path)

    best_params['max_depth'] = int(best_params['max_depth'])

    num_boost_round = int(best_params['num_boost_round'])
    del best_params['num_boost_round']

    dtrain = xgb.DMatrix(X_train, label=y_train)

    model2 = xgb.train(best_params, dtrain, num_boost_round=num_boost_round)

    # Predict on testing and training set
    y_pred = model2.predict(xgb.DMatrix(X_test))
    # y_pred = np.around(y_pred).astype(int)

    y_pred_aux = []
    for value in y_pred:
        if value >= 5:
            value = 5
        elif value <= 0:
            value = 0
        else:
            value = int(round(value))
        y_pred_aux.append(value)

    y_pred = y_pred_aux

    y_train_pred = model2.predict(dtrain)
    # y_train_pred = np.around(y_train_pred).astype(int)

    y_train_pred_aux = []
    for value in y_train_pred:
        if value >= 5:
            value = 5
        elif value <= 0:
            value = 0
        else:
            value = int(round(value))
        y_train_pred_aux.append(value)

    y_train_pred = y_train_pred_aux

    print('-------- Train -----------')
    print(classification_report(y_train, y_train_pred, target_names=labels))
    plot_confusion_matrix(data_true=y_train,
                          data_pred=y_train_pred,
                          classes=labels,
                          title='XGBoost -- Confusion Matrix Training',
                          normalize=True,
                          save_plot=True)

    print('-------- Test -----------')

    print(classification_report(y_test, y_pred, target_names=labels))
    plot_confusion_matrix(data_true=y_test,
                          data_pred=y_pred,
                          classes=labels,
                          title='XGBoost -- Confusion Matrix Test',
                          normalize=True,
                          save_plot=True)


#*******************************************************************************
# Main
#*******************************************************************************
if __name__ == '__main__':
    # Labels

    labels = ['normal', 'imbalance', 'horizontal-misalignment',\
    'vertical-misalignment', 'underhang', 'overhang']

    # Data path parameters
    current_dir = os.path.dirname(os.path.realpath(__file__))
    feat_path = os.path.join(current_dir, "data/data_sep")  # Output feature path

    # Separate train, validation and test
    X_train, y_train, X_test, y_test = open_train_val_data(path=feat_path)

    param_dict = {
        'max_depth': (3, 50),
        'eta': (0.1, 0.3),
        'gamma': (0, 1),
        'colsample_bytree': (0.3, 0.9),
        'num_boost_round': (50, 150)
    }

    train(X_train=X_train,
          y_train=y_train,
          param_to_search=param_dict,
          init_points=3,
          n_iter=5,
          kfold=10)

    # Evaluating traininig with validation data
    evaluation(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, labels=labels)
