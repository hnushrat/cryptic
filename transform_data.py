import pandas as pd
import numpy as np

import random

seed = 42
random.seed(seed)
np.random.seed(seed)

def return_data(filename):
    data = pd.read_csv(filename)

    for i in range(64):
        data[f"pos_{i+1}"] = np.nan


    split = []
    for i in data["Bitstream"].values:
        split.append(list(i))

    data.iloc[:, 3:] = split

    data.drop(["CID", "Bitstream"], axis = 1, inplace = True)

    return data.astype('int')


def return_data_v2(filename):
    data = pd.read_csv(filename)
    
    space_split = []
    for i in data["Bitstream"].values:
        space_split.append(" ".join(list(i)))

    data["space_split"] = space_split

    data.drop(["CID", "Bitstream"], axis = 1, inplace = True)

    return data