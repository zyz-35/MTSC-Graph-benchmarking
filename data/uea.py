#包导入的格式是适配运行根目录的文件，以下语句为了同时适配直接调试本文件
import sys
import os
current_path = os.path.abspath(os.path.dirname(__file__))
parent_path = os.path.abspath(os.path.join(current_path, '..'))
sys.path.append(parent_path)
#=========================================================

from torch.utils.data import Dataset
import numpy as np
from data.config import register_dataset
from data.utils.uealabelmapping import label_mapping

@register_dataset
class UEADataset(Dataset):
    def __init__(self,
                 name: str,
                 path: str,
                 train: bool = False):
        split = "_TRAIN.npz" if train else "_TEST.npz"
        path = os.path.expanduser("~/data/UEA")
        data_path = os.path.join(path, name)
        data_file = os.path.join(data_path, name + split)
        train = np.load(data_file)
        self.X, self.y = train["X"], train["y"]
        self.dict = label_mapping[name]
    
    def num_nodes(self):
        return self.X.shape[1]

    def __len__(self):
        return len(self.y)
        
    def __getitem__(self, index):
        x = self.X[index].astype("float32")
        y = self.dict[self.y[index]]
        return index, x, y

if __name__ == "__main__":
    import os
    from torch.utils.data import DataLoader
    import torch
    name = "DuckDuckGeese"
    path = os.path.expanduser("~/data/UEA")
    traindataset = UEADataset(name, path, True)
    print(traindataset.num_nodes())
    # np.set_printoptions(threshold=np.inf)
    # idx, x, y = traindataset.__getitem__(11)
    # print(x.shape, y)
    # print(traindataset.__getitem__(0)[1].shape)
    # testdataset = UEADataset(name, path, False)
    # trainloader = DataLoader(traindataset, 2)
    # testloader = DataLoader(testdataset, 2)
    # trainX = torch.concat([x for x, _ in trainloader])
    # testX = torch.concat([x for x, _ in testloader])
    # print(trainX.shape)
    # print(testX.shape)