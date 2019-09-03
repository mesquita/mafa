import itertools
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix


def plot_confusion_matrix(data_true,
                          data_pred,
                          classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=None,
                          show_plot=True,
                          save_plot=False):
    """Gera, exibe e salva a matriz de confusão referente as predições de um classificador.

    O comportamento padrão é apenas exibir a matriz de confusão, sem salvá-la nem normalizar suas
    entradas.

    Args:
        data_true (:obj:`array`): Valores alvo (corretos).
        data_pred (:obj:`array`): Valores preditos pelo classificador.
        classes (:obj:`array`, opcional): Lista de rótulos para indexar a matriz.
        normalize (:obj:`bool`, opcional): The second parameter. O padrão é False.
        title (str): Título do plot e da imagem salva. O padrão é 'Confusion matrix'
        cmap (str or :obj:`matplotlib.colors.Colormap`, opcional): Objeto Colormap ou nome colormap
        registrado. Usado para mapear escalares em cores. O padrão é :obj:`plt.cm.Blues`.
        show_plot (:obj:`bool`, opcional): The second parameter. O padrão é True.
        save_plot (:obj:`bool`, opcional): The second parameter. O padrão é False.
    """
    if cmap is None:
        cmap = plt.cm.Blues

    cm = confusion_matrix(data_true, data_pred)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    plt.figure(figsize=(7, 7))
    if show_plot:
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j,
                 i,
                 format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    # plt.show()
    if save_plot:
        current_dir = os.path.dirname(os.path.realpath(__file__))
        path_to_save = os.path.join(current_dir, '..', title)
        plt.savefig(path_to_save, bbox_inches='tight')
