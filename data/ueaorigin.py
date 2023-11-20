#包导入的格式是适配运行根目录的文件，以下语句为了同时适配直接调试本文件
import sys
import os
current_path = os.path.abspath(os.path.dirname(__file__))
parent_path = os.path.abspath(os.path.join(current_path, '..'))
sys.path.append(parent_path)
#=========================================================

from sktime.datasets import load_UCR_UEA_dataset
from torch.utils.data import Dataset
import warnings
import pandas as pd
import numpy as np
from pandas.errors import PerformanceWarning
from data.config import register_dataset
from data.utils.uealabelmapping import label_mapping
warnings.filterwarnings("ignore", category=PerformanceWarning)

irregular = {
    "CharacterTrajectories": 182,
    "InsectWingbeat": 22, # 与官网的csv文件中描述的30不符，且实际训/测比是25000:25000
    "JapaneseVowels": 29,
    "SpokenArabicDigits": 93
}

def load(name, split=None, return_X_y=True, return_type="numpy3d",
         extract_path=None):
    data = load_UCR_UEA_dataset(name, split, return_X_y, return_type,
                                extract_path)
    return data

# @register_dataset
class UEADataset(Dataset):
    def __init__(self,
                 name: str,
                 path: str,
                 train: bool = False):
        split = "train" if train else "test"
        return_type = "numpy3d"
        if name in irregular:
            return_type = None
            self.max_len = irregular[name]
        self.X, self.y = load(
            name, split, return_type=return_type, extract_path=path)
        self.dict = label_mapping[name]
    
    def __len__(self):
        return len(self.y)
        
    def __getitem__(self, index):
        if isinstance(self.X, pd.DataFrame):
            x = self.X.iloc[index]
            x = pd.concat(list(x), axis=1).values
            x = np.nan_to_num(x.astype("float32"))
            x = np.pad(x, ((0, self.max_len - len(x)), (0, 0))).T
        else:
            x = self.X[index].astype("float32")
        y = self.dict[self.y[index]]
        return index, x, y

if __name__ == "__main__":
    import os
    from torch.utils.data import DataLoader
    import torch
    name = "ArticularyWordRecognition"
    path = os.path.expanduser("~/data/UEA")
    traindataset = UEADataset(name, path, True)
    np.set_printoptions(threshold=np.inf)
    print(traindataset.__getitem__(0)[1])
    # testdataset = UEADataset(name, path, False)
    # trainloader = DataLoader(traindataset, 2)
    # testloader = DataLoader(testdataset, 2)
    # trainX = torch.concat([x for x, _ in trainloader])
    # testX = torch.concat([x for x, _ in testloader])
    # print(trainX.shape)
    # print(testX.shape)