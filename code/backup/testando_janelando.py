import os

import numpy as np
import pandas as pd

from utils.win_feats import compute_statistical_features as statfeat

current_dir = os.path.dirname(os.path.realpath(__file__))

sign_nms = ['TCH', 'IAA', 'IRA', 'ITA', 'EAA', 'ERA', 'ETA', 'MIC']

path = os.path.join(current_dir, "data/mafaulda/normal/12.288_0.csv")
df = pd.read_csv(path, header=None, names=sign_nms)

df.index = pd.to_timedelta(df.index, unit='T')
df = df.resample('25T').mean()

janelada = statfeat(data=d''f, window_size=4 * 10**3, center=False)
print(janelada)
print(janelada.columns)
