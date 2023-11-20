import yaml
import os
from pathlib import Path
from config import get_cfg_defaults, convert_to_dict, merge_from_file_withoutsafe

def change(confs, des):
    for f in Path(confs).iterdir():
        cfg = get_cfg_defaults()
        merge_from_file_withoutsafe(cfg, f)
        cfg = convert_to_dict(cfg)
        # 更正数据位置
        cfg["DATASET"]["PARAM"]["path"] = "~/data/UEA"
        # 处理dropout和wd
        # cfg["EXPERIMENT"]["OPTIMIZER"]["PARAM"]["weight_decay"] = 0.0
        # cfg["MODEL"]["PARAM"]["dropout"] = 0.0
        # 处理thred，gat和megat有
        # if ("thred" in cfg["MODEL"]["PARAM"]):
        #     cfg["MODEL"]["PARAM"]["thred"] = 0.2
        # 保存
        new = Path(des, f.name)
        with open(new, "w", encoding="utf-8") as file:
            yaml.dump(cfg, file)

def rename(confs, i):
    for f in Path(confs).iterdir():
        old = f.name
        if "PEMS-SF" in old:
            n, a, g, _, _, _ = old.split("-")
            d = "PEMS-SF.yml"
        else: 
            n, a, g, _, d = old.split("-")
        new = "-".join([n, a, g, str(i), d])
        new = Path(confs, new)
        os.rename(f, new)

if __name__ == "__main__":
    # 最新2001
    i = 2001
    confs = os.path.expanduser("~/testconf")
    des = os.path.expanduser(f"~/testconf")
    # confs = os.path.expanduser(f"~/conf{i}")
    # des = os.path.expanduser(f"~/conf{i}")
    if not os.path.exists(des):
        os.makedirs(des)
    change(confs, des)
    # rename(des, i)