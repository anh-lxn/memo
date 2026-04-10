from __future__ import annotations

import numpy as np
import pandas as pd

from memo.types import NormalizationStats


def min_max_normalize(data: pd.DataFrame):
    data_filtered = data.iloc[:, -8:]
    mins = data_filtered.min()
    maxs = data_filtered.max()
    norm_data_filtered = (data_filtered - mins) / (maxs - mins)
    norm_data_filtered = 2 * norm_data_filtered - 1
    norm_data = data.copy()
    norm_data.iloc[:, -8:] = norm_data_filtered
    stats = NormalizationStats(mins=np.array(mins), maxs=np.array(maxs))
    return norm_data, stats.mins, stats.maxs

