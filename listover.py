from pathlib import Path
from typing import Union

def listover(ready: Union[str, Path], path: Union[str, Path]):
    tasks = set()
    path = Path(path)
    for conf in path.iterdir():
        task = "_".join(conf.name.split("_")[0:-1])
        tasks.add(task)
    ready = Path(ready)
    rtasks = set()
    for conf in ready.iterdir():
        task = "_".join(conf.name.split("_")[0:-1])
        rtasks.add(task)
    over = [task for task in tasks if task not in rtasks]
    for task in over:
        print(task)

if __name__ == "__main__":
    listover("config", "config/over")