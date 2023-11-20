import torch
import time
import numpy as np
import scipy.stats
from scipy.signal import butter, filtfilt
from typing import Optional, Union

def differential_entropy(x: torch.Tensor,
                         fs: int,
                         bands: Optional[list[tuple[float, float]]]
                         ) -> torch.Tensor:
    x = x.cpu().numpy()
    fbands = torch.stack([frequency_band(x, fs, fb) for fb in bands], -1)
    de = scipy.stats.differential_entropy(fbands, axis=-2)
    return torch.tensor(de).type(torch.float)

def frequency_band(x: Union[torch.Tensor, np.ndarray],
                   fs: int,
                   freq_band: tuple[float, float]) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        x = x.cpu().numpy()
    low, high = freq_band
    try:
        b, a = butter(4, [low / (fs / 2), high / (fs / 2)], btype='band')
    except ValueError:
        b, a = butter(4, low / (fs / 2), btype="high")
    filtered_data = filtfilt(b, a, x, axis=-1)
    return torch.tensor(filtered_data.copy()).type(torch.float)

if __name__ == "__main__":
    x = torch.randn(16, 64, 3000)
    end = time.time()
    adj = torch.stack([differential_entropy(g, 1000, [(1, 4), (4, 8), (8, 14), (14, 31), (31, 50)]) for g in x])
    de_time = time.time() - end
    end = time.time()
    adj_batch = differential_entropy(x, 1000, [(1, 4), (4, 8), (8, 14), (14, 31), (31, 50)])
    de_batch_time = time.time() - end
    print(adj.equal(adj_batch))
    print(f"time of de computing: {de_time}")
    print(f"time of de computing: {de_batch_time}")