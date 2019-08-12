import os

import numpy as np
import pandas as pd
from sklearn import preprocessing

folder_path = os.path.dirname(os.path.abspath(__file__))

filepath_faulty = os.path.join(folder_path, 'data', 'imbalance', '10g', '25.6.csv')
filepath_normal = os.path.join(folder_path, 'data', 'normal', '12.288.csv')

df_faulty = pd.read_csv(filepath_faulty)
df_normal = pd.read_csv(filepath_normal)

faulty_col = df_faulty['-0.57341']
normal_col = df_normal
