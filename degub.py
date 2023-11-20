import os
from sktime.datasets import load_UCR_UEA_dataset
import time

path = os.path.expanduser("~/data/UEA")
start = time.time()
train = load_UCR_UEA_dataset("InsectWingbeat", "train", True, extract_path=path)
test = load_UCR_UEA_dataset("InsectWingbeat", "test", True, extract_path=path)
end = time.time()
print(start - end)