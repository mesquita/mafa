import fnmatch  # Filtering filenames
import os
import pickle
import random
from collections import Counter

import numpy as np
import pandas as pd
import xgboost as xgb
from bayes_opt import BayesianOptimization
from bayes_opt.util import Colours
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, classification_report, f1_score
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


def xgb_cv(max_depth, learning_rate, n_estimators, gamma, data, targets):
    clf = xgb.XGBClassifier(max_depth=max_depth,
                            learning_rate=learning_rate,
                            n_estimators=n_estimators,
                            gamma=gamma,
                            random_state=33)

    estimator = Pipeline(steps=[('zcore', StandardScaler()), ('pca', PCA(0.99)), ('clf', clf)])

    cval = cross_val_score(estimator, data, targets, scoring='neg_log_loss', cv=10)
    return cval.mean()


def optimize_xgb(X_train, y_train, param_dict, n_iter=5, nfold=3):
    dtrain = xgb.DMatrix(X_train, label=y_train)

    def xgb_crossval(max_depth, learning_rate, n_estimators, gamma):
        return xgb_cv(max_depth=int(max_depth),
                      learning_rate=learning_rate,
                      n_estimators=int(n_estimators),
                      gamma=gamma,
                      data=X_train,
                      targets=y_train)

    optimizer = BayesianOptimization(xgb_crossval, pbounds=param_dict, verbose=2)

    optimizer.maximize(n_iter=n_iter)

    best_params = optimizer.max['params']

    return best_params


def train(X_train, y_train, param_to_search, n_iter=5, kfold=10, best_param_path=None):
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
        current_dir = os.path.dirname(os.path.realpath(__file__))
        best_param_path = os.path.join(current_dir, "data/best_params/", type_of_data,
                                       "best_params_xgb.pickle")

    best_params = optimize_xgb(X_train=X_train,
                               y_train=y_train,
                               param_dict=param_to_search,
                               n_iter=n_iter,
                               nfold=kfold)
    save_obj(obj=best_params, name=best_param_path)


def save_clf_pred(path, y_pred, y_test):
    path = path.replace("best_params", "clf_pred").replace("pickle", "csv")
    pd.DataFrame.from_dict({'y_pred': y_pred, 'y_test': y_test}).to_csv(path)


def evaluation(X_train, y_train, X_test, y_test, labels, type_of_data='raw', best_param_path=None):
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
        current_dir = os.path.dirname(os.path.realpath(__file__))
        best_param_path = os.path.join(current_dir, "data/best_params/", type_of_data,
                                       "best_params_xgb.pickle")

    best_params = load_obj(best_param_path)

    best_params['max_depth'] = int(best_params['max_depth'])
    best_params['n_estimators'] = int(best_params['n_estimators'])

    clf_best = xgb.XGBClassifier(**best_params)
    clf_best.fit(X_train, y_train)

    y_pred = clf_best.predict(X_test)
    y_train_pred = clf_best.predict(X_train)
    save_clf_pred(best_param_path, y_pred, y_test)
    print('-------- Train -----------')
    print(classification_report(y_train, y_train_pred, target_names=labels))
    plot_confusion_matrix(data_true=y_train,
                          data_pred=y_train_pred,
                          classes=labels,
                          title=(type_of_data + '_XGBoost -- Confusion Matrix Training'),
                          normalize=True,
                          save_plot=True)
    print(f'Accuracy: {accuracy_score(y_train, y_train_pred)}')
    print(f'F1 score [macro]: {f1_score(y_train, y_train_pred, average="macro")}')
    print(f'F1 score [micro]: {f1_score(y_train, y_train_pred, average="micro")}')
    print('-------- Test -----------')

    print(classification_report(y_test, y_pred, target_names=labels))
    plot_confusion_matrix(data_true=y_test,
                          data_pred=y_pred,
                          classes=labels,
                          title=(type_of_data + '_XGBoost -- Confusion Matrix Test'),
                          normalize=True,
                          save_plot=True)
    print(f'Accuracy: {accuracy_score(y_test, y_pred)}')
    print(f'F1 score [macro]: {f1_score(y_test, y_pred, average="macro")}')
    print(f'F1 score [micro]: {f1_score(y_test, y_pred, average="micro")}')


#*******************************************************************************
# Main
#*******************************************************************************
if __name__ == '__main__':
    # Labels
    random.seed(33)

    labels = ['normal', 'imbalance', 'horizontal-misalignment',\
    'vertical-misalignment', 'underhang', 'overhang']

    # Data path parameters
    current_dir = os.path.dirname(os.path.realpath(__file__))
    type_of_data = 'raw'  # 'wavelet' / 'raw' / 'all_feat'
    feat_path = os.path.join(current_dir, 'data', 'data_sep', type_of_data)

    # Separate train, validation and test
    X_train, y_train, X_test, y_test = open_train_val_data(path=feat_path)

    param_dict = {
        'max_depth': (10, 200),
        'learning_rate': (0.1, 0.3),
        'n_estimators': (20, 120),
        'gamma': (0, 1)
    }

    train(X_train=X_train, y_train=y_train, param_to_search=param_dict, n_iter=1, kfold=10)

    # Evaluating traininig with validation data
    evaluation(X_train=X_train,
               y_train=y_train,
               X_test=X_test,
               y_test=y_test,
               type_of_data=type_of_data,
               labels=labels)
