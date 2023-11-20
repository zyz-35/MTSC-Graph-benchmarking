import os
from config import get_cfg_defaults, merge_from_file_withoutsafe
from nn.config import models
from data.config import datasets
from data.utils.adjacency import adj_matrix
from data.utils.node import node_feature
import torch
import argparse
from torch.utils.data import DataLoader
import torch.nn as nn
from utils import train_adj, test_adj, train_node, test_node
from utils import lambda_train, lambda_test
from nn.chebnet import ChebNet
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("--conf_file", help="配置文件")
parser.add_argument("--model_file", help="模型保存位置")
parser.add_argument("--gpu", help="显卡id")

def feature(conf_file: str, model_file: str, gpu: int) -> torch.Tensor:
    
    #==================通用设置====================#

    print("=====> Loading config...")
    cfg = get_cfg_defaults()
    merge_from_file_withoutsafe(cfg, conf_file)
    cfg.SYSTEM.GPU = gpu

    cfg.freeze()


    os.environ["CUDA_VISIBLE_DEVICES"] = cfg.SYSTEM.GPU
    torch.manual_seed(cfg.SYSTEM.SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    #==================读取数据====================#

    Dataset = datasets[cfg.DATASET.CLASS]
    trainset = Dataset(**dict(cfg.DATASET.PARAM), train=True)
    testset = Dataset(**dict(cfg.DATASET.PARAM), train=False)
    trainloader = DataLoader(trainset, 1024, False,
                            num_workers=cfg.SYSTEM.NUM_WORKERS, pin_memory=True)
    testloader = DataLoader(testset, 1024, False,
                            num_workers=cfg.SYSTEM.NUM_WORKERS, pin_memory=True)

    #==================读取模型====================#

    model = models[cfg.MODEL.CLASS](**dict(cfg.MODEL.PARAM))
    cp = torch.load(model_file)
    model.load_state_dict(cp["model_state_dict"])
    model = model.to(device)
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        # 就这一行
        model = nn.DataParallel(model)


    #==================推理过程====================#

    lambda_train.reset()
    lambda_test.reset()
    train_node.reset()
    test_node.reset()
    train_adj.reset()
    test_adj.reset()

    func2path = {
        "complete": "COM",
        "abspearson": "PCC",
        "mutual_information": "MI",
        "phase_locking_value": "PLV",
        "differential_entropy": "DE",
        "power_spectral_density": "PSD"
    }

    datapath = os.path.expanduser(cfg.DATASET.PARAM.path)
    try:
        adj_path = os.path.join(datapath,
                                func2path[cfg.GRAPH.ADJ_MATRIX],
                                f"{cfg.DATASET.PARAM.name}")
    except KeyError:
        adj_path = None
    if adj_path is not None:
        train_adj.set_path(adj_path)
        test_adj.set_path(adj_path)
    adjacency = adj_matrix[cfg.GRAPH.ADJ_MATRIX]

    try:
        node_path = os.path.join(datapath,
                                 func2path[cfg.GRAPH.NODE],
                                 f"{cfg.DATASET.PARAM.name}")
    except KeyError:
        node_path = None
    if node_path is not None:
        train_node.set_path(node_path)
        test_node.set_path(node_path)
    node_feat = node_feature[cfg.GRAPH.NODE]
    
    try:
        lambda_path = os.path.join(datapath,
                                "lambda",
                                func2path[cfg.GRAPH.ADJ_MATRIX],
                                f"{cfg.DATASET.PARAM.name}")
    except KeyError:
        lambda_path = None
    lambda_train.set_path(lambda_path)
    lambda_test.set_path(lambda_path)

    try:
        fs = cfg.DATASET.fs
        bands = cfg.DATASET.bands
    except AttributeError:
        fs = 0
        bands = None
        
    # 定义一个列表储存mlp层之前的特征
    features = []
    origins = []
    labels = []

    # 定义一个函数来处理指定中间特征
    def hook_fn(module, input, output):
        # 根据情况选择取input
        # features.append(input)
        # 还是output
        features.append(output)

    # 注册hook
    # 图网络的readout特征
    # hook_layer = model.mlp.register_forward_hook(hook_fn)
    # 第一层gcn后的特征
    hook_layer = model.convs[1].register_forward_hook(hook_fn)

    # 运行模型
    test_adj.set_adj(adjacency)
    test_node.set_node_feat(node_feat)
    model.eval()
    with torch.no_grad():
        for idx, input, label in testloader:
            if fs != 0 and bands is not None:
                node = test_node.node(input, fs, bands, idx, "_TEST.pt")
            elif fs != 0:
                node = test_node.node(input, fs, None, idx, "_TEST.pt")
            else:
                node = test_node.node(input, None, None, idx, "_TEST.pt")
            adj = test_adj.adj(input, idx, "_TEST.pt")

            input, label = input.to(device), label.to(device)
            adj = adj.to(device)
            node = node.to(device)

            if isinstance(model, ChebNet):
                lambda_max = lambda_test.lambda_max(idx, "_TEST.pt")
                model(adj, node, input, lambda_max)
            else:
                model(adj, node, input)
            origins.append(input)
            labels.append(label)

    # 取出中间特征
    f = features[0].cpu()
    origin = origins[0].cpu()
    label = labels[0].cpu()

    # 移除hook
    hook_layer.remove()

    return origin, f, label

def plot_tsne(origin: torch.Tensor, f: torch.Tensor, label):
    tsne = TSNE(n_components=2, perplexity=12.0, learning_rate=200.0)
    origin = origin.flatten(start_dim=1, end_dim=-1)
    f = f.flatten(start_dim=1, end_dim=-1)
    # f = f.max(dim=1).values
    O = origin.numpy()
    F = f.numpy()
    label = label.numpy()
    # Fit and transform the data
    O_tsne = tsne.fit_transform(O)
    F_tsne = tsne.fit_transform(F)
    fig1, ax1 = plt.subplots()
    fig2, ax2 = plt.subplots()
    ax1.scatter(O_tsne[:, 0], O_tsne[:, 1], c=label, cmap="tab20", marker="o")
    ax2.scatter(F_tsne[:, 0], F_tsne[:, 1], c=label, cmap="tab20", marker="o")
    # plt.show()
    # fig1.savefig("Origin.png")
    fig2.savefig("gcn_1.png")

# python main.py --log_path /raid/zhouyz/log/raw-phase_locking_value-gcn-1/raw-phase_locking_value-gcn-1-UWaveGestureLibrary --conf_file /raid/zhouyz/config/raw-phase_locking_value-gcn-1/raw-phase_locking_value-gcn-1-UWaveGestureLibrary/raw-phase_locking_value-gcn-1-UWaveGestureLibrary.yml --save_path /raid/zhouyz/model/raw-phase_locking_value-gcn-1/raw-phase_locking_value-gcn-1-UWaveGestureLibrary --gpu 0
if __name__ == "__main__":
    args = ["--conf_file", "~/reconf2/raw-complete-gcn-30-ArticularyWordRecognition.yml",
            "--model_file", "~/remodel2/raw-complete-gcn-30-ArticularyWordRecognition-174.pth",
            "--gpu", "1"]
    args = parser.parse_args(args)
    # args = parser.parse_args()
    conf_file = os.path.expanduser(args.conf_file)
    model_file = os.path.expanduser(args.model_file)
    gpu = args.gpu
    o, f, l = feature(conf_file, model_file, gpu)
    plot_tsne(o, f, l)