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


def save_obj(obj, name):
    with open(name, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(name):
    with open(name, 'rb') as f:
        return pickle.load(f)


def open_data(path):
    """Open the csv data.

    Args:
        path (string): string that is the path to csv.

    Returns:
        pandas.DataFrame: dataframe with data from csv.
    """
    data = pd.read_csv(path, index_col='Sample')
    return data


def get_data_and_label(data):
    """Separate the features from labels

    Args:
        data (pandas.DataFrame): dataframe containing both features and labels.
        Labels must be in a column called 'Class', that is the default.

    Returns:
        X, y: dataframe with only features and series with only labels.
    """

    y = data['Class']
    X = data.drop(labels='Class', axis=1)

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


def make_label_distr_df(y, y_train, y_val, y_test, labels):
    """Make pandas.Dataframe with label percentage in each input.

    Args:
        y (pandas.Series/numpy.array): series/array containing all the labels
        y_train (pandas.Series/numpy.array): series/array containing training labels
        y_val (pandas.Series/numpy.array): series/array containing validation labels
        y_test (pandas.Series/numpy.array): series/array containing test labels
        labels (list): list containing labels names

    Returns:
        pandas.DataFrame: dataframe containing ordered label percentage in each case.
    """

    # create empty dataframe
    df = pd.DataFrame(columns=['y', 'y_train', 'y_val', 'y_test'], index=labels)

    # populate with percentage
    df['y'] = compute_label_percentage(y)
    df['y_train'] = compute_label_percentage(y_train)
    df['y_val'] = compute_label_percentage(y_val)
    df['y_test'] = compute_label_percentage(y_test)

    return df


def sep_train_val_test(df, labels):
    """Separtes train, validation and test sets.

    Args:
        df (pandas.DataFrame): dataframe containig all the data and labels.
        labels (list): list containing each label name.

    Returns:
        X_train, y_train, X_val, y_val: dataframes and series containing data and labels.
    """
    # separates features and label
    X, y = get_data_and_label(df)

    # separate train & test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # save the X_test and y_test
    current_dir = os.path.dirname(os.path.realpath(__file__))
    np.savetxt(os.path.join(current_dir, "data/X_test.csv"), X_test, delimiter=",")
    np.savetxt(os.path.join(current_dir, "data/y_test.csv"), y_test, delimiter=",")

    # separate train and validation
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2)

    df = make_label_distr_df(y=y, y_train=y_train, y_val=y_val, y_test=y_test, labels=labels)

    print('Porcentagem das separações:')
    print(df)
    df.to_csv(path_or_buf=os.path.join(current_dir, "data/sep_percen.csv"))

    return X_train, y_train, X_val, y_val


def rfc_cv(n_estimators, min_samples_split, max_features, data, targets):
    """Random Forest cross validation.
    This function will instantiate a random forest classifier with parameters
    n_estimators, min_samples_split, and max_features. Combined with data and
    targets this will in turn be used to perform cross validation. The result
    of cross validation is returned.
    Our goal is to find combinations of n_estimators, min_samples_split, and
    max_features that minimzes the log loss.
    """
    estimator = RandomForestClassifier(n_estimators=n_estimators,
                                       min_samples_split=min_samples_split,
                                       max_features=max_features,
                                       random_state=2)
    cval = cross_val_score(estimator, data, targets, scoring='neg_log_loss', cv=4)
    return cval.mean()


def optimize_rfc(data, targets, pbounds=None):
    """Apply Bayesian Optimization to Random Forest parameters."""
    def rfc_crossval(n_estimators, min_samples_split, max_features):
        """Wrapper of RandomForest cross validation.
        Notice how we ensure n_estimators and min_samples_split are casted
        to integer before we pass them along. Moreover, to avoid max_features
        taking values outside the (0, 1) range, we also ensure it is capped
        accordingly.
        """
        return rfc_cv(
            n_estimators=int(n_estimators),
            min_samples_split=int(min_samples_split),
            max_features=max(min(max_features, 0.999), 1e-3),
            data=data,
            targets=targets,
        )

    optimizer = BayesianOptimization(f=rfc_crossval, pbounds=pbounds, random_state=1234, verbose=2)
    optimizer.maximize(n_iter=2)

    print("Final result:", optimizer.max)
    return optimizer.max


def train(X_train,
          y_train,
          type_param_search='gridsearch',
          param_to_search=None,
          kfold=None,
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

    if param_to_search is None:
        param_to_search = {'n_estimators': [50, 100], 'max_depth': [10, 20, 30]}

    if type_param_search == 'gridsearch':
        # Random Forest classifier
        clf = RandomForestClassifier()

        CV_rfc = GridSearchCV(estimator=clf, param_grid=param_to_search, cv=kfold)
        CV_rfc.fit(X_train, y_train)

        if best_param_path is None:
            best_param_path = "best_params.pickle"
        save_obj(obj=CV_rfc.best_params_, name=best_param_path)

    if type_param_search == 'bayesian':
        param_to_search = {
            "n_estimators": [10, 250],
            "min_samples_split": [2, 25],
            "max_features": [0.1, 0.999],
        }
        best_params = optimize_rfc(data=X_train, targets=y_train, pbounds=param_to_search)

    else:
        raise NotImplementedError


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
    feat_path = os.path.join(current_dir, "data/data.csv")  # Output feature path

    # Open features and label data
    df = open_data(feat_path)

    # Separate train, validation and test
    X_train, y_train, X_val, y_val = sep_train_val_test(df=df, labels=labels)

    # Training
    param_gridsearch = {
        'n_estimators': [100, 200, 500],
        'max_depth': [4, 5, 15, 20, 30, 50],
        'criterion': ['gini', 'entropy']
    }

    param_bay = {
        'n_estimators': [1, 500],
        'max_depth': [4, 50],
    }

    train(X_train=X_train,
          y_train=y_train,
          type_param_search='bayesian',
          param_to_search=param_grid,
          kfold=10)

    # Evaluating traininig with validation data
    evaluation(X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val)