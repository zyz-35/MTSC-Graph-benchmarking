import os
import numpy as np
from sktime.datasets import load_UCR_UEA_dataset


path = os.path.expanduser("~/data/UEA")
train_x, train_y = load_UCR_UEA_dataset("DuckDuckGeese", "train", True, "numpy3d", extract_path=path)
test_x, test_y = load_UCR_UEA_dataset("DuckDuckGeese", "test", True,  "numpy3d", extract_path=path)
np_path = os.path.join(path, "DuckDuckGeese")
np_train = os.path.join(np_path, "DuckDuckGeese" + "_TRAIN.npz")
np_test = os.path.join(np_path, "DuckDuckGeese"  + "_TEST.npz")
train = np.load(np_train)
test = np.load(np_test)
print(np.all(train["X"] == train_x))
print(np.all(test["X"] == test_x))
print(np.all(train["X"] == test["X"]))
print(np.all(train_x == test_x))

# for p in problem:
#     path = os.path.expanduser("~/data/UEA")
#     np_path = os.path.join(path, p)
#     np_train = os.path.join(np_path, p + "_TRAIN.npz")
#     np_test = os.path.join(np_path, p + "_TEST.npz")
#     train = np.load(np_train)
#     test = np.load(np_test)
#     train_x, train_y = train["X"], train["y"]
#     test_x, test_y = test["X"], test["y"]
#     print(p + ":")
#     print("train:")
#     print(train_x.shape, train_y.shape)
#     print("test:")
#     print(test_x.shape, test_y.shape)