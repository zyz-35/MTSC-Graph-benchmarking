from pathlib import Path
import shutil
import os

def chooseconf(logs, confs, des):
    cs = []
    path = Path(logs)
    for f in path.iterdir():
        cs.append(f.stem + ".yml")
    for c in cs:
        src = os.path.join(confs, c)
        shutil.copy(src, des)
    
def rename(confs, i):
    for f in Path(confs).iterdir():
        old = f.name
        if "PEMS-SF" in old:
            n, a, g, i, _, _ = old.split("-")
            d = "PEMS-SF.yml"
        else: 
            n, a, g, i, d = old.split("-")
        new = "-".join([n, a, g, str(i), d])
        new = Path(confs, new)
        os.rename(f, new)

if __name__ == "__main__":
    # path = os.path.expanduser("~/r1log")
    # confs = os.path.expanduser("~/r1conf")
    des = os.path.expanduser("~/conf2001")
    # chooseconf(path, confs, des)
    rename(des, 2001)
    