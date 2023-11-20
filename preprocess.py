import os
from sktime.datasets import load_UCR_UEA_dataset
import time
import pandas as pd
import numpy as np
from ueainfo import problem

irregular = {
    "CharacterTrajectories": 182,
    "InsectWingbeat": 22, # 与官网的csv文件中描述的30不符，且实际训/测比是25000:25000
    "JapaneseVowels": 29,
    "SpokenArabicDigits": 93
}

path = os.path.expanduser("~/data/UEA")
for p in problem:
    if p in irregular:
        return_type = None
        max_len = irregular[p]
    else:
        return_type = "numpy3d"
    train_x, train_y = load_UCR_UEA_dataset(p, "train", True, return_type, extract_path=path)
    if return_type is None:
        tr_list = []
        for idx, row in train_x.iterrows():
            trx = pd.concat(list(row), axis=1).values
            trx = np.nan_to_num(trx.astype("float32"))
            trx = np.pad(trx, ((0, max_len - len(trx)), (0, 0))).T
            tr_list.append(trx)
        tr_np = np.array(tr_list)
    else:
        tr_np = train_x

    test_x, test_y = load_UCR_UEA_dataset(p, "test", True, return_type, extract_path=path)
    if return_type is None:
        te_list = []
        for idx, row in test_x.iterrows():
            x = pd.concat(list(row), axis=1).values
            x = np.nan_to_num(x.astype("float32"))
            x = np.pad(x, ((0, max_len - len(x)), (0, 0))).T
            te_list.append(x)
        te_np = np.array(te_list)
    else:
        te_np = test_x

    train_np = (tr_np, train_y)
    test_np = (te_np, test_y)
    
    np_path = os.path.join(path, p)
    np_train = os.path.join(np_path, p + "_TRAIN")
    np_test = os.path.join(np_path, p + "_TEST")
    np.savez(np_train, X=train_np[0], y=train_np[1])
    np.savez(np_test, X=test_np[0], y=test_np[1])
    print(f"{p} done")