#包导入的格式是适配运行根目录的文件，以下语句为了同时适配直接调试本文件
import sys
import os
current_path = os.path.abspath(os.path.dirname(__file__))
parent_path = os.path.abspath(os.path.join(current_path, '..'))
sys.path.append(parent_path)
#=========================================================

import torch
from torch.utils.data import Dataset
import numpy as np
from data.config import register_dataset
from data.utils.uealabelmapping import label_mapping

abbr = {
    "complete": "COM",
    "abspearson": "PCC",
    "mutual_information": "MI",
    "phase_locking_value": "PLV",
    "differential_entropy": "DE",
    "power_spectral_density": "PSD"
}

@register_dataset
class UEADataset2(Dataset):
    def __init__(self,
                 name: str,
                 path: str,
                 node: str,
                 adj: str,
                 train: bool = False):
        split = "_TRAIN.npz" if train else "_TEST.npz"
        split_pt = "_TRAIN.pt" if train else "_TEST.pt"
        path = os.path.expanduser(path)
        data_path = os.path.join(path, name)
        data_file = os.path.join(data_path, name + split)
        data = np.load(data_file)
        self.X, self.y = data["X"], data["y"]
        self.dict = label_mapping[name]
        self.node = self.get_node(node, path, name, split_pt)
        self.adj = self.get_adj(adj, path, name, split_pt)
        self.lmax = self.get_lmax(adj, path, name, split_pt)
        ...
    
    def get_node(self, node: str, path: str, name: str, split: str):
        if node == "raw":
            res = torch.Tensor(self.X).type(torch.float32)
        else:
            file = os.path.join(path, abbr[node], name+split)
            res = torch.load(file, map_location=torch.device("cpu"))
        return res
    
    def get_adj(self, adj: str, path:str, name: str, split: str):
        num_nodes = self.X[0].shape[0]
        if adj == "identity":
            res = torch.stack([torch.eye(num_nodes),]*len(self.y))
        elif adj == "complete":
            res = torch.ones(len(self.y), num_nodes, num_nodes)
        else:
            file = os.path.join(path, abbr[adj], name+split)
            res = torch.load(file, map_location=torch.device("cpu"))
        return res

    def get_lmax(self, adj: str, path:str, name: str, split: str):
        if adj == "identity":
            res = None
        else:
            file = os.path.join(path, "lambda", abbr[adj], name+split)
            res = torch.load(file, map_location=torch.device("cpu"))
        return res

    def __len__(self):
        return len(self.y)
        
    def __getitem__(self, index):
        x = self.X[index].astype("float32")
        y = self.dict[self.y[index]]
        node = self.node[index].type(torch.float32)
        adj = self.adj[index]
        lmax = self.lmax[index]
        return x, y, node, adj, lmax

if __name__ == "__main__":
    import os
    from torch.utils.data import DataLoader
    import torch
    name = "DuckDuckGeese"
    path = os.path.expanduser("~/data/UEA")
    node = "raw"
    adj = "complete"
    traindataset = UEADataset2(name, path, node, adj, True)
    np.set_printoptions(threshold=np.inf)
    x, y, node, adj, lmax = traindataset.__getitem__(11)
    print(x.shape, y, node.shape, adj.shape, lmax)
    # print(traindataset.__getitem__(0)[1].shape)
    # testdataset = UEADataset(name, path, False)
    # trainloader = DataLoader(traindataset, 2)
    # testloader = DataLoader(testdataset, 2)
    # trainX = torch.concat([x for x, _ in trainloader])
    # testX = torch.concat([x for x, _ in testloader])
    # print(trainX.shape)
    # print(testX.shape)