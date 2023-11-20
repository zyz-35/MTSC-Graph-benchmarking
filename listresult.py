from pathlib import Path
import numpy as np
import pandas as pd
from typing import Union
from decimal import Decimal


def getacc(log: Path) -> float:
    acc = []
    with open(log, "r") as f:
        for line in f:
            idx = line.find("best_accuracy")
            if idx != -1:
                line = line[line.find("]"):]
                for w in line.split():
                    try:
                        acc.append(float(w))
                        break
                    except ValueError:
                        pass
    try:
        acc = np.array(acc).max()
    except ValueError:
        acc = None
    return acc

def printresult(log_path: Path):
    for log in log_path.iterdir():
        acc = getacc(log)
        print(f"{log.stem}: {acc}")

def csvresult(log_path: Path, save_path: Union[Path, str]):
    results = pd.DataFrame(columns=["problem", "node", "adj", "network", "acc"])
    for i, log in enumerate(log_path.iterdir()):
        acc = getacc(log)
        if acc is None:
            continue
        acc = f"{(acc/100):.3f}"
        words = log.stem.split("_")
        raw = [words[-1], words[0], " ".join(words[1:-2]), words[-2], acc]
        results.loc[i] = raw
    results.sort_values(
        by=["problem", "node", "adj", "network"],
        axis=0, ignore_index=True, inplace=True)
    save_path = Path(save_path) if isinstance(save_path, str) else save_path
    results.to_csv(save_path, index=False)

def compresult(log_path: Path, save_path: Union[Path, str]):
    node = ["raw", "psd", "de"]
    adj = ["com", "pcc", "mi", "plv", "diff"]
    net = ["GCN", "ChebNet", "GAT", "STGCN", "MEGAT"]
    columns = ["_".join([v, e, g]) for v in node for e in adj for g in net]
    index = [
        "ArticularyWordRecognition",
        "AtrialFibrillation",
        "BasicMotions",
        "CharacterTrajectories",
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
        "InsectWingbeat",
        "JapaneseVowels",
        "Libras",
        "LSST",
        "MotorImagery",
        "NATOPS",
        "PenDigits",
        "PEMS-SF",
        "Phoneme",
        "RacketSports",
        "SelfRegulationSCP1",
        "SelfRegulationSCP2",
        "SpokenArabicDigits",
        "StandWalkJump",
        "UWaveGestureLibrary"
    ]
    adj = {
        "complete": "com",
        "abspearson": "pcc",
        "mutual_infomation": "mi",
        "phase_locking_value": "plv",
        "diffgraphlearn": "diff"
    }
    results = pd.DataFrame(columns=columns, index=index)
    for log in log_path.iterdir():
        acc = getacc(log)
        if acc is None:
            continue
        acc = Decimal(acc / 100).quantize(
            Decimal("0.001"), rounding="ROUND_HALF_UP")
        words = log.stem.split("_")
        v = words[0]
        e = adj["_".join(words[1:-2])]
        g = words[-2]
        p = words[-1]
        if p == "PhonemeSpectra":
            p = "Phoneme"
        method = "_".join([v, e, g])
        results.loc[p][method] = acc
    results.to_csv(save_path)

if __name__ == "__main__":
    log_path = Path.home() / "log"
    save_path = Path.cwd() / "result71.csv"
    compresult(log_path, save_path)