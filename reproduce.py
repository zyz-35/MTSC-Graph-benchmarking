import torch
from nn.config import models
from data.config import datasets
from data.utils.adjacency import adj_matrix
from data.utils.node import node_feature
from utils import train_adj, test_adj, train_node, test_node
from utils import lambda_train, lambda_test
from train import train
from test import test
from torch.utils.data import DataLoader
from data.uea import UEADataset
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--resume", help="模型保存位置")
parser.add_argument("--name", help="数据集名称")
# parser.add_argument("--gpu", help="显卡id")
args = [
    "--resume", "/data/zhouyz/remodel2/raw-complete-gcn-30-ArticularyWordRecognition-170.pth",
    "--name", "ArticularyWordRecognition"
]

args = parser.parse_args()

resume = args.resume
name = args.name

testset = UEADataset(name=name, path="~/data/UEA", train=False)
testloader = DataLoader(testset, 1024, False,
                        num_workers=4, pin_memory=True)
gt = []
for idx, input, label in testloader: 
    gt.extend(label.tolist())
checkpoint = torch.load(resume)
preds = checkpoint["prediction"]

plt.plot(range(len(gt)), gt, label='ground truth', color="r")
plt.plot(range(len(preds)), preds, label='prediction', color="blue")
# 添加图例
plt.legend()
# 显示图形
plt.show()