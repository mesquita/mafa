import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils.freq_feats import freq_feat

# *******************************************************************************
# Main
# *******************************************************************************

if __name__ == '__main__':

    current_dir = os.path.dirname(os.path.realpath(__file__))

    # Main parameters
    path = os.path.join(current_dir, "data/mafaulda/normal/12.288.csv")
    df = pd.read_csv(path, header=None)
    whole_signal = df[0].to_frame()
    frq_vec = freq_feat(data=whole_signal, which_feat='smallest_freq')
    print(frq_vec)
