# Basic imports
import re
import time
from itertools import product
from re import split

import numpy as np  # Numpy library (numeric algebra)
import pandas as pd  # Pandas library (csv reader, etc...)
import pywt


def truncate_signal(data, level):
    """Trunca o sinal de forma que este seja multiplo de 2**level.

    Args:
        data (pandas DataFrame): sinal a ser truncado
        level (int): nivel de decomposição

    Returns:
        tuple: sinal truncado e número de amostras dropadas
    """

    # primeiro passo: Redimensionar o sinal, dado a nota na documentacao do pywt.swt:
    # The implementation here follows the “algorithm a-trous” and requires that
    # the signal length along the transformed axis be a multiple of 2**level.
    # If this is not the case, the user should pad up to an appropriate size using a function such as numpy.pad.
    # https://pywavelets.readthedocs.io/en/latest/ref/swt-stationary-wavelet-transform.html
    cutting_size = 0
    if data.shape[0] % 2**level != 0:
        # trunk dataframe
        cutting_size = _compute_cutting_size(data, level)
        df_data = data.iloc[:-cutting_size]
    else:
        df_data = data
    return df_data, cutting_size


def cols_namer(cols, wavelet_name, level):
    '''Constrói o nome dos coeficientes da wavelet `wavelet_name` para cada uma das colunas `cols`.

       Os coeficientes seguem a seguinte nomenclatura:
        `'{COL_NAME}_{WAVELET_NAME}_{COEF_NAME}{LEVEL_NB}'`,
       onde `COEF_NAME` pode assumir os valores:
        *`'cA'`: coeficientes de aproximação
        *`'cD'`: coeficientes de detalhes
        ou, no caso da wavelets reduzidas,
        *`'sc'`: coeficientes da escalas
        *`'wv'`: coeficientes da waveltes

    Args:
        cols (list): Lista com os nomes dos
        wavelet_name (string): Nome da wavelet a ser usada
        levels (int): Número de níveis da wavelet

    Returns:
        Lista contendo os nomes a serem usados para os diferentes níves de cada variável

    Examples:
        >>> _cols_namer(['Pressao'], 'bior6', 2)
        ['Pressao_bior6_cA2', 'Pressao_bior6_cD2', 'Pressao_bior6_cA1', 'Pressao_bior6_cD1']

    '''
    # Definindo o nome das colunas

    coefs_names = [(f'cA{n}', f'cD{n}'.format(n)) for n in range(level, 0, -1)]

    coefs_names = list(np.concatenate(coefs_names))

    # nome colunas
    cols_name = []
    for col, coef_name in product(*[cols, coefs_names]):
        cols_name.append(f'{col}_{wavelet_name}_{coef_name}')

    return cols_name


def swt_decomposition(df_data, wavelet, level):
    """Stationary wavelet decomposition

    Arguments:
        df_data {pandas DataFrame} -- dataframe com os dados
        wavelet {Wavelet obj ou nome} -- wavelet a ser utilizada
        level {int} -- número de decomposições

    Returns:
        numpy array -- array contendo coeficientes de aproximação e detalhe
        em mesma ordem que a função wavedec do pywt: [(cAn, cDn), ..., (cA2, cD2), (cA1, cD1)]
    """

    decomposition = []

    for col in df_data.columns:
        # extraindo coeficientes wavelets por tag
        swt_coeffs = pywt.swt(df_data[col], wavelet, level=level, axis=0)

        decomposition_tag = np.concatenate(swt_coeffs)
        decomposition.append(decomposition_tag)

    return np.concatenate(decomposition)


def fill_data(data, padding_size):
    """Insere `padding_size` (com os mesmos valores da última linha) linhas extras (axis=0) ao Dataframe `data`

    Args:
        data (pandas DataFrame): data
        padding_size (int): número de linhas extras a ser adicionadas

    Returns:
        pandas DataFrame: Dataframe com as linhas adicionadas
    """

    cols_name = data.columns

    # repetir a última linha "padding_size" vezes
    vec = np.array(data.iloc[-1]).reshape(1, -1)
    data_to_append = np.repeat(vec, padding_size, axis=0)

    df_to_append = pd.DataFrame(data_to_append, columns=cols_name)

    data = data.append(df_to_append, ignore_index=True)

    return data
