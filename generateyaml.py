import yaml
from ueainfo import tasks, problem
import os
from config import get_cfg_defaults, convert_to_dict
from itertools import product

irregular = {
    "CharacterTrajectories": 182,
    "InsectWingbeat": 22, # 与官网的csv文件中描述的30不符，且实际训/测比是12500:12500
    "JapaneseVowels": 29,
    "SpokenArabicDigits": 93
}

# datasets = problem
# datasets = ["ArticularyWordRecognition", "BasicMotions", "Cricket", "DuckDuckGeese", "EigenWorms", "FaceDetection", "FingerMovements", "HandMovementDirection", "Heartbeat", "LSST", "MotorImagery", "NATOPS", "PEMS-SF", "PhonemeSpectra", "RacketSports", "SelfRegulationSCP1", "SelfRegulationSCP2"]
datasets = ["PenDigits"]
# gnns = ["gat-1"]
gnns = ["megat-1", "megat-2", "megat-3"]
# nodes = ["differential_entropy", "power_spectral_density", "raw"]
nodes = ["raw"]
# adjs = ["mutual_information", "abspearson", "phase_locking_value",
#         "complete", "diffgraphlearn"]
adjs = ["mutual_information"]

def generateyml(config_path, dataset, node, adj, gnn):
    if not os.path.exists(config_path):
        os.makedirs(config_path)
    conf = get_cfg_defaults()
    conf.merge_from_file(os.path.join("config/template", gnn + ".yml"))
    conf = convert_to_dict(conf)
    conf["GRAPH"]["NODE"] = node
    if adj == "diffgraphlearn":
        conf["MODEL"]["PARAM"]["graphlearn"] = True
        conf["GRAPH"]["ADJ_MATRIX"] = "identity"
    else:
        conf["GRAPH"]["ADJ_MATRIX"] = adj
    idx = problem.index(dataset)
    p, l, c, b, fs, bands  = tasks[idx]
    if (fs == 0) and (node in ["differential_entropy", "power_spectral_density"]):
        # comb = [dataset, node, adj, gnn]
        # print(f"This conbination won't be used: {comb}")
        return
    if ("stgcn" in gnn) and (node in ["differential_entropy", "power_spectral_density"]):
        # comb = [dataset, node, adj, gnn]
        # print(f"This conbination won't be used: {comb}")
        return
    # if (gnn == "stgcn") and (p == "PenDigits"):
    #     conf["MODEL"]["PARAM"]["k"] = 2
    large_graph = (dataset == "DuckDuckGeese") | (dataset == "PEMS-SF")
    if large_graph and (adj == "diffgraphlearn"):
        if "megat" in gnn:
            if dataset == "DuckDuckGeese":
                conf["EXPERIMENT"]["BATCH_SIZE"] = 4
            else:
                conf["EXPERIMENT"]["BATCH_SIZE"] = 8
        else:
            conf["EXPERIMENT"]["BATCH_SIZE"] = 16
        # if gnn == "stgcn":
        #     conf["EXPERIMENT"]["BATCH_SIZE"] = 8
        # else:
        #     conf["EXPERIMENT"]["BATCH_SIZE"] = 16
    elif large_graph and ("megat" in gnn):
        if dataset == "DuckDuckGeese":
            conf["EXPERIMENT"]["BATCH_SIZE"] = 4
        else:
            conf["EXPERIMENT"]["BATCH_SIZE"] = 16
    # 特殊情况使用
    # if large_graph and ("stgcn" in gnn):
    #     conf["EXPERIMENT"]["BATCH_SIZE"] = 32
    conf["DATASET"]["PARAM"]["name"] = p
    conf["DATASET"]["fs"] = fs
    conf["DATASET"]["bands"] = bands
    conf["MODEL"]["PARAM"]["n_classes"] = c
    conf["MODEL"]["PARAM"]["len"] = l
    if node == "raw":
        conf["MODEL"]["PARAM"]["in_dim"] = l
    elif node == "differential_entropy":
        conf["MODEL"]["PARAM"]["in_dim"] = len(bands)
    elif node == "power_spectral_density":
        conf["MODEL"]["PARAM"]["in_dim"] = int(l // 2)
    
    if conf["MODEL"]["PARAM"]["graphlearn"]:
        adj = "diffgraphlearn"
    else:
        adj = conf["GRAPH"]["ADJ_MATRIX"]

    gat_r_pc_sl = ["PEMS-SF", "PhonemeSpectra"]
    if node == "raw" and adj == "abspearson" and "gat" == gnn[:3] and dataset in gat_r_pc_sl:
        conf["MODEL"]["PARAM"]["self_loop"] = True
    gat_r_m_sl = ["PEMS-SF", "PhonemeSpectra"]
    if node == "raw" and adj == "mutual_information" and "gat" == gnn[:3] and dataset in gat_r_m_sl:
        conf["MODEL"]["PARAM"]["self_loop"] = True
    gat_r_a_sl = ["ArticularyWordRecognition", "BasicMotions", "Cricket", "DuckDuckGeese", "EigenWorms", "FaceDetection", "FingerMovements", "HandMovementDirection", "Heartbeat", "LSST", "MotorImagery", "NATOPS", "PEMS-SF", "PhonemeSpectra", "RacketSports", "SelfRegulationSCP1", "SelfRegulationSCP2"]
    if node == "raw" and adj == "diffgraphlearn" and "gat" == gnn[:3] and dataset in gat_r_a_sl:
        conf["MODEL"]["PARAM"]["self_loop"] = True
    gat_d_a_sl = gat_r_a_sl
    if node == "differential_entropy" and adj == "diffgraphlearn" and "gat" == gnn[:3] and dataset in gat_d_a_sl:
        conf["MODEL"]["PARAM"]["self_loop"] = True
    gat_ps_a_sl = gat_r_a_sl
    if node == "power_spectral_density" and adj == "diffgraphlearn" and "gat" == gnn[:3] and dataset in gat_ps_a_sl:
        conf["MODEL"]["PARAM"]["self_loop"] = True
    me_r_pc_sl = ["PEMS-SF", "PhonemeSpectra"]
    if node == "raw" and adj == "abspearson" and "megat" in gnn and dataset in me_r_pc_sl:
        conf["MODEL"]["PARAM"]["self_loop"] = True
    me_r_m_sl = ["PEMS-SF", "PhonemeSpectra"]
    if node == "raw" and adj == "mutual_information" and "megat" in gnn and dataset in me_r_m_sl:
        conf["MODEL"]["PARAM"]["self_loop"] = True
    me_r_a_sl = problem
    if adj == "diffgraphlearn" and "megat" in gnn and dataset in me_r_a_sl:
        conf["MODEL"]["PARAM"]["self_loop"] = True
    # approach_name = node + "-" + adj + "-" + gnn
    # approach_fold = os.path.join(config_path, approach_name)
    # dataset_name = approach_name + "-" + dataset
    # dataset_fold = os.path.join(approach_fold, dataset_name)
    # if not os.path.exists(dataset_fold):
    #     os.makedirs(dataset_fold)
    conf_name = node + "-" + adj \
                + "-" + gnn + "-" + dataset + ".yml"
    conf_file = os.path.join(config_path, conf_name)
    with open(conf_file, "w", encoding="utf-8") as file:
        yaml.dump(conf, file)

if __name__ == "__main__":
    print("The conbinations of these will be generated:")
    print(f"datasets:\n{datasets}")
    print(f"node features:\n{nodes}")
    print(f"adjacencies:\n{adjs}")
    print(f"gnns:\n{gnns}")
    while True:
        flag = input("Generate?[y/[n]]:") or "n"
        if flag == "N" or flag == "n":
            print("Exit")
            break
        elif flag == "Y" or flag == "y":
            for comb in product(datasets, nodes, adjs, gnns):
                generateyml("/data/zhouyz/reconf", *comb)
                # generateyml("config/init", *comb)
            print("Done!")
            break
        else:
            print("Invalid input!")