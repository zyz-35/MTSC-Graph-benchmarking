import numpy as np
import torch
import torch.nn as nn
import logging
from torch.utils.data import DataLoader
import argparse
import os
import torch.optim
import torch.optim.lr_scheduler as lr_scheduler
import datetime
from data.uea import UEADataset
from nn.resnet import ResNet

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", help="数据集名称")
parser.add_argument("--gpu", help="显卡id")

torch.manual_seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def main(args):
    now = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f"[{now}] Start training!")
    # ================读取命令行参数=======================
    dataset_name = args.dataset
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ================读取数据============================
    path = os.path.expanduser("~/data/UEA")   
    trainset = UEADataset(dataset_name, path, True)
    testset = UEADataset(dataset_name, path, False)
    trainloader = DataLoader(trainset, 16, True, num_workers=4, pin_memory=True)
    testloader = DataLoader(testset, 1024, False, num_workers=4, pin_memory=True)

    # =================定义模型===========================
    num_dim = trainset.X.shape[1]
    num_class = len(np.unique(trainset.y))
    model = ResNet(num_dim, num_class)
    model = model.to(device)

    # ==================训练策略==========================
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer,
                                               mode="min",
                                               factor=0.5,
                                               patience=50,
                                               min_lr=0.0001)

    for epoch in range(1500):
        # =================训练==========================
        print(f"==========Epoch{epoch}==========")
        model.train()
        for i, (idx, input, label) in enumerate(trainloader):
            input, label = input.to(device), label.to(device)
            output = model(input)
            loss = loss_fn(output, label)
            loss.backward()
            optimizer.step()

        #===================评估训练集======================
        model.eval()
        total_samples = 0
        train_correct = 0
        train_loss = 0
        preds = []
        with torch.no_grad():
            for i, (idx, input, label) in enumerate(trainloader):
                input, label = input.to(device), label.to(device)
                output = model(input)
                loss = loss_fn(output, label)
                pred = output.argmax(dim=1, keepdim=True)
                preds.extend(pred.squeeze().tolist())
                train_correct += pred.eq(label.view_as(pred)).sum().item()
                train_loss += loss.item() * input.shape[0]
                total_samples += input.shape[0]
        train_loss /= total_samples
        train_accuracy = 100. * train_correct / total_samples
        # 按照训练集的损失函数来判断收敛
        scheduler.step(train_loss)
        print("Train set: Average loss: {:.4f}, "\
                    "Accuracy: {}/{} ({:.2f}%)".format(
            train_loss, train_correct, total_samples, train_accuracy))

        # ==========================评估测试集===========================
        total_samples = 0
        test_loss = 0
        test_correct = 0
        preds = []
        with torch.no_grad():
            for idx, input, label in testloader:
                input, label = input.to(device), label.to(device)
                output = model(input)
                loss = loss_fn(output, label)
                pred = output.argmax(dim=1, keepdim=True)
                preds.extend(pred.squeeze().tolist())
                test_correct += pred.eq(label.view_as(pred)).sum().item()
                test_loss += loss.item() * input.shape[0]
                total_samples += input.shape[0]
        test_loss /= total_samples
        test_accuracy = 100. * test_correct / total_samples
        print("Test set: Average loss: {:.4f},\
              Accuracy: {}/{} ({:.2f}%) ".format(
            test_loss, test_correct, total_samples, test_accuracy))
    
    now = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f"[{now}] Done!")

if __name__ == "__main__":
    # argv = [
    #     "--dataset", "ArticularyWordRecognition",
    #     "--gpu", "0"
    # ]
    args = parser.parse_args()
    main(args)