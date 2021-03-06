import fnmatch  # Filtering filenames
import os
import pickle
import random
from collections import Counter

import numpy as np
import pandas as pd
from bayes_opt import BayesianOptimization
from bayes_opt.util import Colours
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.ensemble import RandomForestClassifier
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


def rfc_cv(n_estimators, max_depth, min_samples_split, data, targets):
    """Random Forest cross validation.
    This function will instantiate a random forest classifier with parameters
    n_estimators, min_samples_split, and max_features. Combined with data and
    targets this will in turn be used to perform cross validation. The result
    of cross validation is returned.
    Our goal is to find combinations of n_estimators, min_samples_split, and
    max_features that minimzes the log loss.

    Args:
        n_estimators ([type]): [description]
        min_samples_split ([type]): [description]
        max_features ([type]): [description]
        data ([type]): [description]
        targets ([type]): [description]

    Returns:
        [type]: [description]
    """

    clf = RandomForestClassifier(n_estimators=n_estimators,
                                 max_depth=max_depth,
                                 min_samples_split=min_samples_split,
                                 random_state=33)

    estimator = Pipeline(steps=[('zcore', StandardScaler()), ('pca', PCA(0.99)), ('clf', clf)])

    cval = cross_val_score(estimator, data, targets, scoring='neg_log_loss', cv=10)
    return cval.mean()


def optimize_rfc(data, targets, pbounds=None, n_iter=2):
    """Apply Bayesian Optimization to Random Forest parameters."""
    def rfc_crossval(n_estimators, max_depth, min_samples_split):
        """Wrapper of RandomForest cross validation.
        Notice how we ensure n_estimators and min_samples_split are casted
        to integer before we pass them along.
        """
        return rfc_cv(
            n_estimators=int(n_estimators),
            max_depth=int(max_depth),
            min_samples_split=int(min_samples_split),
            data=data,
            targets=targets,
        )

    optimizer = BayesianOptimization(f=rfc_crossval, pbounds=pbounds, verbose=2)
    optimizer.maximize(n_iter=n_iter)

    print("Final result:", optimizer.max)
    return optimizer.max['params']


def train(X_train,
          y_train,
          type_of_data='raw',
          type_param_search='gridsearch',
          param_to_search=None,
          kfold=None,
          n_iter_bay=None,
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
        current_dir = os.path.dirname(os.path.realpath(__file__))
        best_param_path = os.path.join(current_dir, "data/best_params/", type_of_data,
                                       "best_params_rf.pickle")

    if param_to_search is None:
        param_to_search = {'n_estimators': [10, 100], 'max_depth': [10, 30]}

    if type_param_search == 'gridsearch':
        # Random Forest classifier
        clf = RandomForestClassifier()

        CV_rfc = GridSearchCV(estimator=clf, param_grid=param_to_search, cv=kfold)
        CV_rfc.fit(X_train, y_train)

        # Saving dict with best_params
        save_obj(obj=CV_rfc.best_params_, name=best_param_path)

    elif type_param_search == 'bayesian':
        if n_iter_bay is None:
            n_iter_bay = 10
        best_params = optimize_rfc(data=X_train,
                                   targets=y_train,
                                   pbounds=param_to_search,
                                   n_iter=n_iter_bay)
        best_params = {k: int(round(v)) for k, v in best_params.items()}
        save_obj(obj=best_params, name=best_param_path)

    else:
        raise NotImplementedError


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
                                       "best_params_rf.pickle")

    best_params = load_obj(best_param_path)

    clf_best = RandomForestClassifier(**best_params)
    clf_best.fit(X_train, y_train)

    y_train_pred = clf_best.predict(X_train)
    y_pred = clf_best.predict(X_test)
    # save_clf_pred(best_param_path, y_pred, y_test)

    print('-------- Train -----------')
    print(classification_report(y_train, y_train_pred, target_names=labels))
    plot_confusion_matrix(data_true=y_train,
                          data_pred=y_train_pred,
                          classes=labels,
                          title=(type_of_data + '_RF -- Confusion Matrix Training'),
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
                          title=(type_of_data + '_RF -- Confusion Matrix Test'),
                          normalize=False,
                          save_plot=True)
    print(f'Accuracy: {accuracy_score(y_test, y_pred)}')
    print(f'F1 score [macro]: {f1_score(y_test, y_pred, average="macro")}')
    print(f'F1 score [micro]: {f1_score(y_test, y_pred, average="micro")}')


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
    feat_path = os.path.join(current_dir, 'data', 'data_sep', type_of_data)

    # Separate train, validation and test
    X_train, y_train, X_test, y_test = open_train_val_data(path=feat_path)

    # Training
    param_gridsearch = {
        'n_estimators': [100, 200, 500],
        'max_depth': [4, 5, 15, 20, 30, 50],
        'criterion': ['gini', 'entropy']
    }

    param_bay = {"n_estimators": [20, 200], 'max_depth': [10, 200], "min_samples_split": [2, 5]}

    # train(X_train=X_train,
    #       y_train=y_train,
    #       type_param_search='bayesian',
    #       param_to_search=param_bay,
    #       type_of_data=type_of_data,
    #       kfold=10,
    #       n_iter_bay=1)

    # Evaluating traininig with validation data
    evaluation(X_train=X_train,
               y_train=y_train,
               X_test=X_test,
               y_test=y_test,
               labels=labels,
               type_of_data=type_of_data)
