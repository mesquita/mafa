import os

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import (GridSearchCV, cross_val_score, train_test_split)

from utils.confusionmatrix import plot_confusion_matrix


def open_data(path):
    data = pd.read_csv(path, index_col='Sample')
    return data


def sep_feat_label(data):

    y = data['Class']
    X = data.drop(labels='Class', axis=1)

    return X, y


def train(path, kfold=None):
    # Labels
    labels = ['normal', 'imbalance', 'horizontal-misalignment',\
    'vertical-misalignment', 'underhang', 'overhang']

    # Open features and label data
    df = open_data(path)

    # Separates features and label
    X, y = sep_feat_label(df)

    # Separate train & test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

    # Random Forest classifier
    clf = RandomForestClassifier(n_estimators=100, max_depth=30, random_state=0)

    param_grid = {
        'n_estimators': [200, 500],
        'max_features': ['auto', 'sqrt', 'log2'],
        'max_depth': [4, 5, 6, 7, 8],
        'criterion': ['gini', 'entropy']
    }
    CV_rfc = GridSearchCV(estimator=clf, param_grid=param_grid, cv=kfold)

    CV_rfc.fit(X_train, y_train)
    print(CV_rfc.best_params_)

    y_pred = clf.predict(X_test)

    print(classification_report(y_test, y_pred, target_names=labels))
    # plot_confusion_matrix(data_true=y_test, data_pred=y_pred, classes=labels, normalize=True)
    feat_import = clf.feature_importances_

    # else:
    #     scores = cross_val_score(clf, X, y, cv=kfold)
    #     print(scores)
    #     print(f'Média: {np.mean(scores)}')

    return print('oi')


#*******************************************************************************
# Main
#*******************************************************************************

if __name__ == '__main__':

    current_dir = os.path.dirname(os.path.realpath(__file__))

    # Main parameters
    feat_path = os.path.join(current_dir, "data/data_original_code.csv")  # Output feature path

    train(path=feat_path, kfold=10)
