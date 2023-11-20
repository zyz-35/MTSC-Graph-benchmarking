from data.utils.adjacency import mutual_information, phase_locking_value, abspearson, complete
from sktime.datasets import load_UCR_UEA_dataset
from data.uea import UEADataset
import torch
from torch.utils.data import DataLoader
import os
import sys
from pathlib import Path
from typing import Union, Optional
from ueainfo import problem

def com_with_print(clip, x, i):
    print(f"computing complete of sample {i} in {clip}")
    return complete(x)

def pcc_with_print(clip, x, i):
    print(f"computing pcc of sample {i} in {clip}")
    return abspearson(x)

def mi_with_print(clip, x, i):
    print(f"computing mi of sample {i} in {clip}")
    return mutual_information(x)

def plv_with_print(clip, x, i):
    print(f"computing plv of sample {i} in {clip}")
    return phase_locking_value(x)

def generate_pcc(name: str, data_path: str, save_path: str) -> None:
    trainset = UEADataset(name, data_path, True)
    trainloader = DataLoader(trainset, trainset.X.shape[0])
    for idx, X, y in trainloader:
        X = X.cuda()
        train_pcc = \
            torch.stack([pcc_with_print("train", x, i) for i, x in enumerate(X)])
    save_train = save_path + "_TRAIN.pt"
    torch.save(train_pcc, save_train)
    testset = UEADataset(name, data_path, False)
    testloader = DataLoader(testset, testset.X.shape[0])
    for idx, X, y in testloader:
        X = X.cuda()
        test_pcc = \
            torch.stack([pcc_with_print("test", x, i) for i, x in enumerate(X)])
    save_test = save_path + "_TEST.pt"
    torch.save(test_pcc, save_test)

def generate_mi(name: str, data_path: str, save_path: str) -> None:
    trainset = UEADataset(name, data_path, True)
    trainloader = DataLoader(trainset, trainset.X.shape[0])
    for idx, X, y in trainloader:
        # X = X.cuda()
        train_mi = \
            torch.stack([mi_with_print("train", x, i) for i, x in enumerate(X)])
    save_train = save_path + "_TRAIN.pt"
    torch.save(train_mi, save_train)
    testset = UEADataset(name, data_path, False)
    testloader = DataLoader(testset, testset.X.shape[0])
    for idx, X, y in testloader:
        # X = X.cuda()
        test_mi = \
            torch.stack([mi_with_print("test", x, i) for i, x in enumerate(X)])
    save_test = save_path + "_TEST.pt"
    torch.save(test_mi, save_test)

def generate_mi_train(name: str, data_path: str, save_path: str) -> None:
    trainset = UEADataset(name, data_path, True)
    trainloader = DataLoader(trainset, trainset.X.shape[0])
    for idx, X, y in trainloader:
        # X = X.cuda()
        train_mi = \
            torch.stack([mi_with_print("train", x, i) for i, x in enumerate(X)])
    save_train = save_path + "_TRAIN.pt"
    torch.save(train_mi, save_train)

def generate_mi_test(name: str, data_path: str, save_path: str) -> None:
    testset = UEADataset(name, data_path, False)
    testloader = DataLoader(testset, testset.X.shape[0])
    for idx, X, y in testloader:
        # X = X.cuda()
        test_mi = \
            torch.stack([mi_with_print("test", x, i) for i, x in enumerate(X)])
    save_test = save_path + "_TEST.pt"
    torch.save(test_mi, save_test)

def generate_plv(name: str, data_path: str, save_path: str) -> None:
    trainset = UEADataset(name, data_path, True)
    trainloader = DataLoader(trainset, trainset.X.shape[0])
    for idx, X, y in trainloader:
        train_plv = \
            torch.stack([plv_with_print("train", x, i) for i, x in enumerate(X)])
    save_train = save_path + "_TRAIN.pt"
    torch.save(train_plv, save_train)
    testset = UEADataset(name, data_path, False)
    testloader = DataLoader(testset, testset.X.shape[0])
    for idx, X, y in testloader:
        test_plv = \
            torch.stack([plv_with_print("test", x, i) for i, x in enumerate(X)])
    save_test = save_path + "_TEST.pt"
    torch.save(test_plv, save_test)

def compute_lambda_max(adj_path: Union[str, Path],
                       save_path: Union[str, Path],
                       adj: Optional[torch.Tensor] = None) -> None:
    if adj is None:
        A = torch.load(adj_path)
    else:
        A = adj
    batch_size, num_nodes, _ = A.shape

    in_degree = 1 / A.sum(dim=2).clamp(min=1).sqrt()
    D_invsqrt = torch.diag_embed(in_degree)
    I = torch.eye(num_nodes).unsqueeze(0).repeat(batch_size, 1, 1).to(A)
    L = I - D_invsqrt @ A @ D_invsqrt

    lambda_ = torch.linalg.eig(L)[0].real
    lambda_max = lambda_.max(-1)[0].view(batch_size, 1, 1)
    torch.save(lambda_max.squeeze(), save_path)   

def generateadj():
    os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[2]
    name = sys.argv[1]
    func = {
        "MI": generate_mi,
        "PLV": generate_plv
    }
    adj_func = func[sys.argv[3]]
    path = os.path.expanduser("~/data/UEA")
    save_path = os.path.join(path, sys.argv[3], name)
    adj_func(name, path, save_path)

def mi_lambda(name: str):
    adj_train = Path.home() / f"data/UEA/MI/{name}_TRAIN.pt"
    save_train = Path.home() / f"data/UEA/lambda/MI/{adj_train.name}"
    compute_lambda_max(adj_train, save_train)
    adj_test = Path.home() / f"data/UEA/MI/{name}_TEST.pt"
    save_test = Path.home() / f"data/UEA/lambda/MI/{adj_test.name}"
    compute_lambda_max(adj_test, save_test)

def cmi_lambda(name: str):
    # adj_train = Path.home() / f"data/UEA/CMI/{name}_TRAIN.pt"
    adj_train = Path(f"/data1/zhouyz/CMI/{name}_TRAIN.pt")
    save_train = Path.home() / f"data/UEA/lambda/CMI/{adj_train.name}"
    compute_lambda_max(adj_train, save_train)
    # adj_test = Path.home() / f"data/UEA/CMI/{name}_TEST.pt"
    adj_test = Path(f"/data1/zhouyz/CMI/{name}_TEST.pt")
    save_test = Path.home() / f"data/UEA/lambda/CMI/{adj_test.name}"
    compute_lambda_max(adj_test, save_test)

def plv_lambda():
    adj_plv = Path.home() / "data/UEA/PLV"
    for plv in adj_plv.iterdir():
        save_path = Path.home() / f"data/UEA/lambda/PLV/{plv.name}"
        compute_lambda_max(plv, save_path)

def pcc_lambda(name: str, data_path: str, save_path: str) -> None:
    trainset = UEADataset(name, data_path, True)
    trainloader = DataLoader(trainset, trainset.X.shape[0])
    for idx, X, y in trainloader:
        train_pcc = \
            torch.stack([pcc_with_print("train", x, i) for i, x in enumerate(X)])
    save_train = save_path + "_TRAIN.pt"
    compute_lambda_max("", save_train, train_pcc)
    testset = UEADataset(name, data_path, False)
    testloader = DataLoader(testset, testset.X.shape[0])
    for idx, X, y in testloader:
        test_pcc = \
            torch.stack([pcc_with_print("test", x, i) for i, x in enumerate(X)])
    save_test = save_path + "_TEST.pt"
    compute_lambda_max("", save_test, test_pcc)

def com_lambda(name: str, data_path: str, save_path: str) -> None:
    trainset = UEADataset(name, data_path, True)
    trainloader = DataLoader(trainset, trainset.X.shape[0])
    for idx, X, y in trainloader:
        train_com = \
            torch.stack([com_with_print("train", x, i) for i, x in enumerate(X)])
    save_train = save_path + "_TRAIN.pt"
    compute_lambda_max("", save_train, train_com)
    testset = UEADataset(name, data_path, False)
    testloader = DataLoader(testset, testset.X.shape[0])
    for idx, X, y in testloader:
        test_com = \
            torch.stack([com_with_print("test", x, i) for i, x in enumerate(X)])
    save_test = save_path + "_TEST.pt"
    compute_lambda_max("", save_test, test_com)

if __name__ == "__main__":
    # sys.argv = [
    #     sys.argv[0],
    #     "ArticularyWordRecognition",
    #     "2",
    #     "PCC"
    # ]
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    done = [
        "ArticularyWordRecognition",
        "AtrialFibrillation",
        "BasicMotions",
        # "CharacterTrajectories",
        "Cricket",
        "DuckDuckGeese",
        "EigenWorms",
        "Epilepsy",
        "EthanolConcentration",
        "ERing",
        "FaceDetection",
        "FingerMovements",
        "HandMovementDirection",
        "Handwriting",
        "Heartbeat",
        # "InsectWingbeat",
        # "JapaneseVowels",
        "Libras",
        "LSST",
        "PEMS-SF",
    ]
    # name = "PEMS-SF"
    # path = os.path.expanduser("~/data/UEA")
    # save_path = os.path.expanduser(f"/data1/zhouyz/CMI/{name}")
    # generate_mi_test(name, path, save_path)
    # generate_mi_train("PEMS-SF", path, save_path)
    for p in problem:
        # if p in done:
        #     continue
        if p != "PEMS-SF":
            continue
        name = p
        print(f"Generate the max lambda of {p}")
        path = os.path.expanduser("/data1/zhouyz")
        save_path = os.path.join(path, "CMI", name)
        cmi_lambda(name)
        # generate_mi(name, path, save_path)