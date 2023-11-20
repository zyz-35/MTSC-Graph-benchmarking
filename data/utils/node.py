import torch
import numpy as np
import scipy.stats
from scipy.signal import butter, filtfilt
from typing import Optional

node_feature = {}
def register_node(func):
    node_feature[func.__name__] = func
    return func

@register_node
def raw(x: torch.Tensor, *args) -> torch.Tensor:
    return x

@register_node
def power_spectral_density(x: torch.Tensor, fs: int, *args) -> torch.Tensor:
    x = x.numpy()
    block_size = len(x[0])
    spec = np.fft.fft(x, axis=-1)
    psd = 2 * np.abs(spec)**2 / (block_size * fs)
    return torch.tensor(psd[:, :block_size//2]).type(torch.float)

@register_node
def differential_entropy(x: torch.Tensor,
                         fs: int,
                         bands: Optional[list[tuple[float, float]]]
                         ) -> torch.Tensor:
    x = x.numpy()
    fbands = torch.stack([frequency_band(x, fs, fb) for fb in bands], -1)
    de = scipy.stats.differential_entropy(fbands, axis=-2)
    return torch.tensor(de).type(torch.float)

def frequency_band(x: torch.Tensor,
                   fs: int,
                   freq_band: tuple[float, float]) -> torch.Tensor:
    # x = x.numpy()
    low, high = freq_band
    try:
        b, a = butter(4, [low / (fs / 2), high / (fs / 2)], btype='band')
    except ValueError:
        b, a = butter(4, low / (fs / 2), btype="high")
    filtered_data = filtfilt(b, a, x, axis=-1)
    return torch.tensor(filtered_data.copy()).type(torch.float)


if __name__ == "__main__":
    bands = [(1, 4), (4, 8), (8, 14), (14, 31), (31, 50)]
    x = np.random.randn(32, 144, 62)
    x = torch.from_numpy(x)
    x = differential_entropy(x, 250, bands)
    print(x.shape)