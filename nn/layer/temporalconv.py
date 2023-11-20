import torch
import torch.nn as nn

class TemporalConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size)
        self.conv2 = nn.Conv1d(in_channels, out_channels, kernel_size)
        self.conv3 = nn.Conv1d(in_channels, out_channels, kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, num_nodes, time_points, feats = x.shape
        x = x.permute(0, 1, 3, 2)
        x = x.contiguous().view(-1, feats, time_points)
        p = self.conv1(x)
        q = torch.sigmoid(self.conv2(x))
        pq = p * q
        h = torch.relu(pq + self.conv3(x))
        h = h.contiguous().view(batch_size, num_nodes, h.shape[-2], h.shape[-1])
        h = h.permute(0, 1, 3, 2)
        return h

class TemporalMaxPool(nn.Module):
    def __init__(self, output_size: int = 1):
        super().__init__()
        self.pool = nn.AdaptiveMaxPool1d(output_size)

    def forward(self, x: torch.Tensor):
        batch_size, num_nodes, time_points, feats = x.shape
        x = x.permute(0, 1, 3, 2)
        x = x.contiguous().view(-1, feats, time_points)
        x = self.pool(x)
        x = x.contiguous().view(batch_size, num_nodes, feats, -1)
        x = x.permute(0, 1, 3, 2)
        return x