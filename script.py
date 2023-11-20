﻿from main import main
import argparse
from pathlib import Path
import os

parser = argparse.ArgumentParser()
parser.add_argument("--log_path", help="日志保存位置")
parser.add_argument("--conf_path", help="配置文件路径")
parser.add_argument("--save_path", help="模型保存位置")
parser.add_argument("--pattern", help="模型保存位置")
parser.add_argument("--gpu", help="显卡id")

# python script.py --log_path /raid/zhouyz/log/round1-lr --conf_path /raid/zhouyz/config/round1-lr --save_path /raid/zhouyz/model/round1-lr --gpu 0

# argv = ["--log_path", "~/log2001", "--conf_path",
#         "~/conf2001", "--save_path", "~/model2001", "--pattern=-gcn-",
#         "--gpu", "0"]
args = parser.parse_args()
log_path = os.path.expanduser(args.log_path)
conf_path = os.path.expanduser(args.conf_path)
save_path = os.path.expanduser(args.save_path)
pattern = args.pattern
gpu = args.gpu

if not os.path.exists(log_path):
    os.makedirs(log_path)
if not os.path.exists(save_path):
    os.makedirs(save_path)

# done_conf = []
e_confs = {}
for conf_file in Path(conf_path).iterdir():
    if not pattern in conf_file.name:
        continue
    if conf_file.is_dir():
        continue
    if Path(log_path, conf_file.stem+".log").exists():
        continue
    # main(log_path, conf_file, save_path, gpu)
    print(conf_file.stem)
    try:
        main(log_path, conf_file, save_path, gpu)
        # done_conf.append(conf_file.stem)
    except KeyboardInterrupt:
        raise
    except Exception as e:
        e_confs[conf_file.stem] = str(e)
        log = Path(log_path, conf_file.stem+".log")
        if log.exists():
            log.unlink()
        # print("Done:")
        # print(done_conf)
        # raise
if len(e_confs) == 0:
    print("All done!")
else:
    print(e_confs)