from config import get_cfg_defaults, merge_from_file_withoutsafe, convert_to_dict
from pathlib import Path
import os
import yaml


def generatevar(save_path, conf_file, key, val):
    cfg = get_cfg_defaults()
    merge_from_file_withoutsafe(cfg, conf_file)
    cfg = convert_to_dict(cfg)
    for v in val:
        if len(key) == 3:
            k = key[2]
            cfg[key[0]][key[1]][key[2]] = v
        if len(key) == 4:
            k = key[3]
            cfg[key[0]][key[1]][key[2]][key[3]] = v
        
        varpost = "_" + k + str(v)
        conf_name = Path(conf_file).stem + varpost + ".yml"
        var_file = os.path.join(save_path, conf_name)
        with open(var_file, "w", encoding="utf-8") as file:
            yaml.dump(cfg, file)
        
if __name__ == "__main__":
    key = ["EXPERIMENT", "OPTIMIZER", "PARAM", "lr"]
    val = [0.1, 0.01, 0.001]
    save_path = "/raid/zhouyz/config/round1-lr"
    conf_path = "/raid/zhouyz/config/init"
    for conf_file in Path(conf_path).iterdir():
        if conf_file.is_dir():
            continue
        generatevar(save_path, conf_file, key, val)