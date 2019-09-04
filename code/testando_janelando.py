import os

import numpy as np
import pandas as pd
from scipy.fftpack import fft

from utils.win_stat_feat import compute_statistical_features as statfeat

mock_dict = {'a': [1, -1, 1, -1], 'b': [1, 1, 1, 1]}

df = pd.DataFrame.from_dict(mock_dict)

janelada = statfeat(data=df, window_size=2, center=False)

print(janelada)

pontos = np.linspace(-np.pi, np.pi, 2001)
seno = np.sin(pontos)
freq_seno = fft(seno)

another_dict = {'seno': seno}

another_df = pd.DataFrame.from_dict(another_dict)

janela = another_df.rolling(2, center=False)
# isso não dá certo pq o rolling.apply tem que devolver 1 elemento
freq = janela.apply(func=fft, axis=1, raw=False)
print(freq)
