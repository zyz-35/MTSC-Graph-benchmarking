#包导入的格式是适配运行根目录的文件，以下语句为了同时适配直接调试本文件
import sys
import os
current_path = os.path.abspath(os.path.dirname(__file__))
parent_path = os.path.abspath(os.path.join(current_path, '..'))
sys.path.append(parent_path)
#=========================================================

import torch
import torch.nn as nn
from nn.layer.megatconv import MEGATConv
from nn.layer.diffgraphlearn import DiffGraphLearn
from nn.layer.graph_edge_model import GEM
from typing import Callable, Optional
from nn.config import register_model

class MEGATLayer(nn.Module):
    def __init__(self,
                 in_feat: int,
                 out_feat: int,
                 num_heads: int,
                 dropout: float,
                 thred: float,
                 residual: bool = False,
                 bias: bool = True,
                 activation: Optional[Callable] = nn.ELU()) -> None:
        super(MEGATLayer, self).__init__()
        self.gconv = MEGATConv(in_feat,
                             out_feat // num_heads,
                             num_heads,
                             dropout,
                             thred,
                             residual=residual,
                             bias=bias,
                             activation=activation)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, adj, x, e):
        x, e = self.gconv(adj, x, e)
        return self.dropout(x), e
        

@register_model
class MEGAT1(nn.Module):
    def __init__(self,
                 len: int,
                 in_dim: int,
                 hidden_dim: int,
                 t_embedding: int,
                 mlp_dim: int,
                 num_heads: int,
                 thred: float,
                 n_classes: int,
                 n_layers: int = 1,
                 dropout: float = 0.5,
                 num_nodes: int = 0,
                 residual: bool = False,
                 readout: str = "sum",
                 graphlearn: bool = False,
                 self_loop: bool = False) -> None:
        super(MEGAT1, self).__init__()
        if graphlearn:
            self.graphlearn = DiffGraphLearn(len)
        else:
            self.graphlearn = None
        # self.embedding = nn.Linear(in_dim, hidden_dim)
        if t_embedding > 0:
            self.t_embed = nn.Linear(in_dim, t_embedding)
            in_dim = t_embedding
        else:
            self.t_embed = nn.Identity()
        # if in_dim == len:
        #     self.t_embed = nn.Linear(in_dim, t_embedding)
        # else:
        #     self.t_embed = nn.Identity()
        # self.norm = nn.BatchNorm1d(hidden_dim)
        # self.edge_extractor = GEM(in_dim)
        self.edge_extractor = GEM(in_dim)
        
        convs = [MEGATLayer(in_dim,
                            hidden_dim,
                            num_heads,
                            dropout,
                            thred,
                            residual)]
        for _ in range(1, n_layers):
            convs.append(MEGATLayer(
                hidden_dim, hidden_dim, num_heads, dropout, thred, residual))
        self.convs = nn.ModuleList(convs)
        # convs = [MEGATLayer(in_dim,
        #                   hidden_dim,
        #                   num_heads,
        #                   dropout,
        #                   thred)
        #             for _ in range(n_layers-1)]
        # self.convs = nn.ModuleList(convs)
        # self.convs.append(
        #     MEGATLayer(hidden_dim, out_dim, num_heads, dropout, thred))
        # self.mlp = nn.Sequential(
        #     nn.Linear(out_dim, mlp_dim),
        #     # nn.BatchNorm1d(mlp_dim),
        #     nn.ReLU(),
        #     nn.Linear(mlp_dim, mlp_dim),
        #     # nn.BatchNorm1d(mlp_dim),
        #     nn.ReLU(),
        #     nn.Linear(mlp_dim, n_classes)
        # )
        self.mlp = nn.Linear(hidden_dim, n_classes)
        self.readout = readout
        self.self_loop = self_loop

    def forward(self, adj: torch.Tensor, x: torch.Tensor,
                raw: torch.Tensor) -> torch.Tensor:
        if self.graphlearn is not None:
            adj = self.graphlearn(raw)
        # x = self.embedding(x)
        if self.self_loop:
            adj_d = adj.diagonal(dim1=-2, dim2=-1).diag_embed().to(adj)
            adj = adj - adj_d + torch.eye(adj.shape[-1]).to(adj)

        # b, v, t = x.shape
        # x = self.norm(x.view(b*v, t)).view(b, v, t)
        # x = torch.relu(x)
        x = self.t_embed(x)
        e = self.edge_extractor(x)
        e = torch.relu(e)

        for conv in self.convs:
            x, e = conv(adj, x, e)
        # # x = torch.sum(x, dim=-2)
        # x = torch.mean(x, dim=-2)
        if self.readout == "sum":
            x = torch.sum(x, dim=-2)
        elif self.readout == "mean":
            x = torch.mean(x, dim=-2)
        
        x = self.mlp(x)
        return x

if __name__ == "__main__":
    x = torch.randn(2, 1345, 270)
    adj = torch.randn(2, 1345, 1345)
    gt = torch.randint(0, 1, (2,))
    adj = (adj > 0).type(torch.float)
    # model = MEGAT(32, 16, 8, 2, 2, 0.8, 2, 1, 0.5)
    critic = torch.nn.CrossEntropyLoss()
    model = MEGAT1(270, 270, 64, 0, 2, 1, 0.8, 2, 2, 0.5, True, "mean")
    print(model)
    y = model(adj, x, x)
    print(y.shape)
    loss = critic(y, gt)
    loss.backword()
    pass