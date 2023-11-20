import os
os.environ["OMP_NUM_THREADS"] = "1"
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
import csv
import shutil
from pathlib import Path
import argparse
from typing import Callable
import optuna
from optuna.trial import Trial
from optuna.logging import _get_library_root_logger
from torch.utils.data import DataLoader
import torch.nn as nn
from utils import train, test, train_adj, test_adj

parser = argparse.ArgumentParser()
parser.add_argument("--log_path", help="日志保存位置")
parser.add_argument("--task_type", help="任务类型，用于匹配配置文件")
parser.add_argument("--gpu", type=int, help="显卡id")

def tune(log_path: str, conf_file: str, gpu: str) -> Callable:
    #==================通用设置====================#

    print("=====> Loading config...")
    end = time.time()
    cfg = get_cfg_defaults()
    # cfg.merge_from_file(conf_file)
    merge_from_file_withoutsafe(cfg, conf_file)
    cfg.SYSTEM.GPU = gpu
    load_config_time = time.time() - end
    print("Config loading time: {:.4f}s".format(load_config_time))

    os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg.SYSTEM.GPU)
    torch.manual_seed(cfg.SYSTEM.SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    task_name = os.path.splitext(os.path.basename(conf_file))[0]
    save_file = os.path.join(cfg.EXPERIMENT.SAVE_PATH, task_name+".pth")

    logger = _get_library_root_logger()
    for handler in logger.handlers:
        if isinstance(handler, logging.FileHandler):
            logger.removeHandler(handler)
    logger.setLevel(logging.DEBUG)
    log_file = os.path.join(log_path, task_name+".log")
    file_handler = logging.FileHandler(log_file)
    formatter = logging.Formatter("[%(levelname)1.1s %(asctime)s] %(message)s")
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.INFO)
    logger.addHandler(file_handler)

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

    def objective(trial: Trial) -> float:
        
        # 需要搜索的超参
        cfg.MODEL.PARAM.dropout = \
            trial.suggest_float("dropout", 0.2, 0.5, step=0.1)
        cfg.MODEL.PARAM.n_layers = trial.suggest_int("n_layers", 1, 2)
        cfg.EXPERIMENT.OPTIMIZER.PARAM.lr = \
            10**trial.suggest_int("lr", -5, -2)
        cfg.EXPERIMENT.OPTIMIZER.PARAM.momentum = \
            trial.suggest_float("momentum", 0, 1.0, step=0.01)
        cfg.EXPERIMENT.OPTIMIZER.PARAM.weight_decay = \
            trial.suggest_float("weight_decay", 0, 1.0, step=0.05)

        # cfg.freeze()

        #==================定义模型====================#

        logger.info("=====> Loading model...")
        end = time.time()
        model = models[cfg.MODEL.CLASS](**dict(cfg.MODEL.PARAM)).to(device)
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

        train_adj.reset()
        test_adj.reset()
        func2path = {
            "mutual_infomation": "MI",
            "phase_locking_value": "PLV"
        }
        try:
            adj_path = os.path.join(cfg.DATASET.PARAM.path,
                                    func2path[cfg.GRAPH.ADJ_MATRIX],
                                    f"{cfg.DATASET.PARAM.name}")
        except KeyError:
            adj_path = None
        if adj_path is not None:
            train_adj.set_path(adj_path)
            test_adj.set_path(adj_path)
        adjacency = adj_matrix[cfg.GRAPH.ADJ_MATRIX]
        node_feat = node_feature[cfg.GRAPH.NODE]
        try:
            fs = cfg.DATASET.fs
            bands = cfg.DATASET.bands
        except AttributeError:
            fs = 0
            bands = None
        best_accuracy = 0.
        best_epoch = 0
        logger.info(f"Experement settting:\n"
                    f"model:\n{model}\n"
                    f"loss_fn:\n{loss_fn}\n"
                    f"optimizer:\n{type(optimizer).__name__} "
                    f"{optimizer.state_dict()}\n"
                    f"scheduler:\n{type(scheduler).__name__} "
                    f"{scheduler.state_dict()}\n")
        for epoch in range(cfg.EXPERIMENT.EPOCHS):
            end = time.time()
            print(f"==========Epoch{epoch}==========")
            train(trainloader, model, loss_fn, optimizer, scheduler, epoch,
                  logger, device, adjacency, node_feat, fs, bands)
            _, accuracy = test(testloader, model, loss_fn, logger, device,
                                adjacency, node_feat, fs, bands)
            logger.info(f"Epoch time: {time.time() - end:.4f}\n")
            
            if accuracy > best_accuracy:
                checkpoint = {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                }
                torch.save(checkpoint, save_file)
                best_accuracy = accuracy
                best_epoch = epoch

            # 发现这一组超参不符合某些条件后直接中断
            trial.report(accuracy, epoch)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

        logger.info("best_accuracy: {:.2f} at epoch {}\n".format(best_accuracy,
                                                                 best_epoch))

        return best_accuracy
    return objective

if __name__ == "__main__":
    args = parser.parse_args()
    log_path = args.log_path
    task_type = args.task_type
    gpu = args.gpu
    # log_path = "/raid/zhouyz/log"
    # task_type = "raw"
    # gpu = 7
    # for conf_file in list(Path("config/").glob(f"{task_type}*.yml")):
    for conf_file in Path("config/").iterdir():
        if conf_file.name.find(f"{task_type}") == -1:
            continue
        print(f"config file: {conf_file.name}")
        study = optuna.create_study(direction="maximize")
        objective = tune(log_path, conf_file, gpu)
        study.optimize(objective, n_trials=10)
        shutil.move(str(conf_file), f"config/over/{conf_file.name}")
        # best_result = study.best_trial.value
        # with open("results.csv", mode="a", newline="") as file:
        #     fieldnames = ["node", "adj_matrix", "model", "dataset", "result"]
        #     writer = csv.DictWriter(file,fieldnames=fieldnames)
        #     if file.tell() == 0:
        #         writer.writeheader()
        #     conf_name = os.path.splitext(os.path.basename(conf_file))[0]
        #     fields = conf_name.split("_")
        #     row = {
        #         "node": fields[0],
        #         "adj_matrix": fields[1],
        #         "model": fields[2],
        #         "dataset": fields[3],
        #         "result": best_result,
        #     }
        #     writer.writerow(row)