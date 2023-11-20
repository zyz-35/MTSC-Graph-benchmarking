import os
from config import get_cfg_defaults, merge_from_file_withoutsafe
from nn.config import models
from data.config import datasets
from optim.optimizer import optimizers
from optim.schedular import schedulers
from data.utils.adjacency import adj_matrix
from data.utils.node import node_feature
import torch
import logging
import time
import argparse
from typing import Callable
from torch.utils.data import DataLoader
import torch.nn as nn
from utils import train_adj, test_adj, train_node, test_node
from utils import lambda_train, lambda_test
from train import train
from test import test

parser = argparse.ArgumentParser()
parser.add_argument("--log_path", help="日志保存位置")
parser.add_argument("--conf_file", help="配置文件")
parser.add_argument("--save_path", help="模型保存位置")
parser.add_argument("--gpu", help="显卡id")

def main(log_path: str, conf_file: str, save_path: str, gpu: int) -> Callable:
    
    #==================通用设置====================#

    print("=====> Loading config...")
    end = time.time()
    cfg = get_cfg_defaults()
    merge_from_file_withoutsafe(cfg, conf_file)
    cfg.SYSTEM.GPU = gpu

    cfg.freeze()
    load_config_time = time.time() - end
    print("Config loading time: {:.4f}s".format(load_config_time))


    os.environ["CUDA_VISIBLE_DEVICES"] = cfg.SYSTEM.GPU
    torch.manual_seed(cfg.SYSTEM.SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    task_name = os.path.splitext(os.path.basename(conf_file))[0]
    save_file = os.path.join(save_path, task_name+".pth")
    save_file_train = os.path.join(save_path, task_name+"_tr"+".pth")
    save_file_loss = os.path.join(save_path, task_name+"_loss"+".pth")
    save_file_trloss = os.path.join(save_path, task_name+"_trloss"+".pth")

    logger = logging.getLogger(task_name)
    if not logger.handlers:
        logger.setLevel(logging.DEBUG)
        formatter = logging.Formatter("[%(levelname)1.1s %(asctime)s] %(message)s")
        log_file = os.path.join(log_path, task_name+".log")
        file_handler = logging.FileHandler(log_file, "w")
        file_handler.setFormatter(formatter)
        file_handler.setLevel(logging.INFO)
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        logger.addHandler(stream_handler)

    #==================读取数据====================#

    Dataset = datasets[cfg.DATASET.CLASS]
    logger.info("=====> Preparing data...")
    end = time.time()
    trainset = Dataset(**dict(cfg.DATASET.PARAM), train=True)
    testset = Dataset(**dict(cfg.DATASET.PARAM), train=False)
    trainloader = DataLoader(trainset, cfg.EXPERIMENT.BATCH_SIZE, True,
                            num_workers=cfg.SYSTEM.NUM_WORKERS, pin_memory=True)
    testloader = DataLoader(testset, cfg.EXPERIMENT.BATCH_SIZE, False,
                            num_workers=cfg.SYSTEM.NUM_WORKERS, pin_memory=True)
    prepare_time = time.time() - end
    logger.info("Data preparing time: {:.4f}s".format(prepare_time))

    #==================定义模型====================#

    logger.info("=====> Loading model...")
    end = time.time()
    model = models[cfg.MODEL.CLASS](**dict(cfg.MODEL.PARAM)).to(device)
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        # 就这一行
        model = nn.DataParallel(model)
    load_model_time = time.time() - end
    logger.info("Model loading time: {:.4f}s".format(load_model_time))

    #==================训练设置====================#

    loss_fn = nn.CrossEntropyLoss()
    optimizer = optimizers[cfg.EXPERIMENT.OPTIMIZER.CLASS]
    optimizer = optimizer(model.parameters(),
                        **dict(cfg.EXPERIMENT.OPTIMIZER.PARAM))
    scheduler = schedulers[cfg.EXPERIMENT.SCHEDULER.CLASS]
    scheduler = scheduler(optimizer, **dict(cfg.EXPERIMENT.SCHEDULER.PARAM))

    #==================训练过程====================#

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
    best_accuracy = 0.
    best_train_accuracy = 0.
    best_train_loss = float("inf")
    best_loss = float("inf")
    best_epoch = 0
    logger.info(f"Experement setting:\n"
                f"model:\n{model}\n"
                f"loss_fn:\n{loss_fn}\n"
                f"optimizer:\n{type(optimizer).__name__} "
                f"{optimizer.state_dict()}\n"
                f"scheduler:\n{type(scheduler).__name__} "
                f"{scheduler.state_dict()}\n")
    for epoch in range(cfg.EXPERIMENT.EPOCHS):
        end = time.time()
        logger.info(f"==========Epoch{epoch}==========")
        lr = optimizer.state_dict()["param_groups"][0]["lr"]
        logger.info(f"Learning rate: {lr}")
        train_loss, train_accuracy, train_pred = \
        train(trainloader, model, loss_fn, optimizer, scheduler, epoch, logger,
            device, adjacency, node_feat, fs, bands)
        loss, accuracy, pred = test(testloader, model, loss_fn, logger, device,
                            adjacency, node_feat, fs, bands)
        if train_loss != train_loss:
            logger.error(f"Loss is {loss}!")
            return -1
        scheduler.step(train_loss)
        logger.info(f"Epoch time: {time.time() - end:.4f}\n")
        
        # megat的EW保存的模型过大了
        # checkpoint = {
        #     "model_state_dict": model.state_dict(),
        #     "optimizer_state_dict": optimizer.state_dict(),
        #     "scheduler_state_dict": scheduler.state_dict(),
        #     "prediction": pred
        # }
        # save_file = os.path.join(save_path, task_name + "-" + str(epoch) +".pth")
        # torch.save(checkpoint, save_file)
        # if accuracy >= best_accuracy:
        #     checkpoint = {
        #         "model_state_dict": model.state_dict(),
        #         "optimizer_state_dict": optimizer.state_dict(),
        #         "scheduler_state_dict": scheduler.state_dict(),
        #         "prediction": pred
        #     }
        #     torch.save(checkpoint, save_file)
        #     best_accuracy = accuracy
        #     best_epoch = epoch
        
        # if train_accuracy >= best_train_accuracy:
        #     checkpoint_tr = {
        #         "model_state_dict": model.state_dict(),
        #         "optimizer_state_dict": optimizer.state_dict(),
        #         "scheduler_state_dict": scheduler.state_dict(),
        #         "prediction": pred
        #     }
        #     best_train_accuracy = accuracy
        #     torch.save(checkpoint_tr, save_file_train)

        # if loss <= best_loss:
        #     checkpoint_loss = {
        #         "model_state_dict": model.state_dict(),
        #         "optimizer_state_dict": optimizer.state_dict(),
        #         "scheduler_state_dict": scheduler.state_dict(),
        #         "prediction": pred
        #     }
        #     best_loss = loss
        #     torch.save(checkpoint_loss, save_file_loss)

        # if train_loss <= best_train_loss:
        #     checkpoint_trloss = {
        #         "model_state_dict": model.state_dict(),
        #         "optimizer_state_dict": optimizer.state_dict(),
        #         "scheduler_state_dict": scheduler.state_dict(),
        #         "prediction": pred
        #     }
        #     best_train_loss = loss
        #     torch.save(checkpoint_trloss, save_file_trloss)

    # logger.info(f"best_accuracy: {best_accuracy:.2f} at epoch {best_epoch}\n")
    logger.info(f"{task_name} done!")


    return best_accuracy

# python main.py --log_path /raid/zhouyz/log/raw-phase_locking_value-gcn-1/raw-phase_locking_value-gcn-1-UWaveGestureLibrary --conf_file /raid/zhouyz/config/raw-phase_locking_value-gcn-1/raw-phase_locking_value-gcn-1-UWaveGestureLibrary/raw-phase_locking_value-gcn-1-UWaveGestureLibrary.yml --save_path /raid/zhouyz/model/raw-phase_locking_value-gcn-1/raw-phase_locking_value-gcn-1-UWaveGestureLibrary --gpu 0
if __name__ == "__main__":
    args = ["--log_path", "~/melog",
            "--conf_file", "~/meconf/diffDDG/raw-diffgraphlearn-megat-1-DuckDuckGeese.yml",
            "--save_path", "~/memodel",
            "--gpu", "0,1,2,3"]
    args = parser.parse_args(args)
    # args = parser.parse_args()
    log_path = os.path.expanduser(args.log_path)
    conf_file = os.path.expanduser(args.conf_file)
    save_path = os.path.expanduser(args.save_path)
    gpu = args.gpu
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    main(log_path, conf_file, save_path, gpu)