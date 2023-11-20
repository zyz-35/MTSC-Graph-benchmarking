from data.utils.node import power_spectral_density, differential_entropy
import os
import numpy as np
import torch

# [problem, serieslen, n_classes, batch_sizes, fs, bands]
tasks = [
    ["ArticularyWordRecognition", 144, 25, 16, 200, [(1, 4), (4, 8), (8, 14), (14, 31), (31, 50)]],
    ["AtrialFibrillation", 640, 3, 1, 128, [(1, 4), (4, 8), (8, 14), (14, 31), (31, 50)]],
    ["BasicMotions", 100, 4, 2, 10, [(1, 4), (4, 5)]],
    ["CharacterTrajectories", 182, 20, 64, 200, [(1, 4), (4, 8), (8, 14), (14, 31), (31, 50)]],
    ["Cricket", 1197, 12, 8, 184, [(1, 4), (4, 8), (8, 14), (14, 31), (31, 50)]],
    ["DuckDuckGeese", 270, 5, 4, 0, []],
    ["EigenWorms", 17984, 5, 8, 0, []],
    ["Epilepsy", 206, 4, 8, 16, [(1, 4), (4, 8)]],
    ["EthanolConcentration", 1751, 4, 16, 0, []],
    ["ERing", 65, 6, 2, 0, []],
    ["FaceDetection", 62, 2, 256, 250, [(1, 4), (4, 8), (8, 14), (14, 31), (31, 50)]],
    ["FingerMovements", 50, 2, 16, 100, [(1, 4), (4, 8), (8, 14), (14, 31), (31, 50)]],
    ["HandMovementDirection", 400, 4, 8, 0, []],
    ["Handwriting", 152, 26, 8, 0, []],
    ["Heartbeat", 405, 2, 8, 0, []],
    ["InsectWingbeat", 22, 10, 1024, 0, []],
    ["JapaneseVowels", 29, 9, 16, 0, []],
    ["Libras", 45, 15, 8, 0, []],
    ["LSST", 36, 14, 128, 0, []],
    ["MotorImagery", 3000, 2, 16, 1000, [(1, 4), (4, 8), (8, 14), (14, 31), (31, 50)]],
    ["NATOPS", 51, 6, 8, 0, []],
    ["PenDigits", 8, 10, 256, 0, []],
    ["PEMS-SF", 144, 7, 16, 0, []],
    ["PhonemeSpectra", 217, 39, 128, 0, []],
    ["RacketSports", 30, 4, 8, 10, [(1, 4), (4, 5)]],
    ["SelfRegulationSCP1", 896, 2, 16, 256, [(1, 4), (4, 8), (8, 14), (14, 31), (31, 50)]],
    ["SelfRegulationSCP2", 1152, 2, 8, 256, [(1, 4), (4, 8), (8, 14), (14, 31), (31, 50)]],
    ["SpokenArabicDigits", 93, 10, 256, 0, []],
    ["StandWalkJump", 2500, 3, 1, 500, [(1, 4), (4, 8), (8, 14), (14, 31), (31, 50)]],
    ["UWaveGestureLibrary", 315, 8, 8, 0, []],
]

def generate_de(data_path, node_path):
    for p in tasks:
        if len(p[5]) == 0:
            continue
        np_path = os.path.join(data_path, p[0])
        np_train = os.path.join(np_path, p[0] + "_TRAIN.npz")
        np_test = os.path.join(np_path, p[0] + "_TEST.npz")
        train = np.load(np_train)
        test = np.load(np_test)
        train_x = train["X"]
        test_x = test["X"]
        fs = p[4]
        bands = p[5]
        train_node = [differential_entropy(x, fs, bands) for x in train_x]
        train_node = torch.stack(train_node)
        test_node = [differential_entropy(x, fs, bands) for x in test_x]
        test_node = torch.stack(test_node)
        save_train = os.path.join(node_path, p[0] + "_TRAIN.pt")
        save_test = os.path.join(node_path, p[0] + "_TEST.pt")
        torch.save(train_node, save_train)
        torch.save(test_node, save_test)
        print(f"{p[0]} done!")

def generate_psd(data_path, node_path):
    for p in tasks:
        if p[4] == 0:
            continue
        np_path = os.path.join(data_path, p[0])
        np_train = os.path.join(np_path, p[0] + "_TRAIN.npz")
        np_test = os.path.join(np_path, p[0] + "_TEST.npz")
        train = np.load(np_train)
        test = np.load(np_test)
        train_x = train["X"]
        test_x = test["X"]
        fs = p[4]
        train_node = [power_spectral_density(x, fs) for x in train_x]
        train_node = torch.stack(train_node)
        test_node = [power_spectral_density(x, fs) for x in test_x]
        test_node = torch.stack(test_node)
        save_train = os.path.join(node_path, p[0] + "_TRAIN.pt")
        save_test = os.path.join(node_path, p[0] + "_TEST.pt")
        torch.save(train_node, save_train)
        torch.save(test_node, save_test)
        print(f"{p[0]} done!")

if __name__ == "__main__":
    path = os.path.expanduser("~/data/UEA")
    de_path = os.path.join(path, "DE")
    psd_path = os.path.join(path, "PSD")
    generate_de(path, de_path)
    generate_psd(path, psd_path)