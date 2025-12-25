import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from copy import deepcopy
from torch_scatter import scatter_add
from torch.nn import ModuleList
from torch_geometric.nn import GCNConv


# -------------------------------------------------------------
# 正規化隣接行列 A_hat (= D^{-1/2} A D^{-1/2}) を返す
# -------------------------------------------------------------
def get_symmetrically_normalized_adjacency(edge_index, n_nodes):
    edge_weight = torch.ones((edge_index.size(1),), device=edge_index.device)
    row, col = edge_index[0], edge_index[1]

    deg = scatter_add(edge_weight, col, dim=0, dim_size=n_nodes)
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float("inf"), 0.0)

    norm_weight = deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]
    return edge_index, norm_weight


# -------------------------------------------------------------
# PaGNN 本体
# -------------------------------------------------------------
class PaGNN(torch.nn.Module):
    def __init__(
        self,
        in_channels,
        hidden_channels,
        out_channels,
        num_layers=2,
        dropout=0.5,
        mask=None,
    ):
        super().__init__()

        self.initial_conv = PaGNNConv(in_channels, hidden_channels)

        self.gc_layers = ModuleList([
            GCNConv(hidden_channels, hidden_channels)
            for _ in range(max(0, num_layers - 2))
        ])

        self.output_conv = GCNConv(hidden_channels, out_channels)

        self.dropout = dropout
        self._mask = mask

    def forward(self, x, edge_index, adj=None, adj2=None, x_label=None, edge_weight=None, **kwargs):
        mask = self._prepare_mask(x, self._mask)

        x = self.initial_conv(x, edge_index, mask)

        for conv in self.gc_layers:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.output_conv(x, edge_index)
        return x

    @staticmethod
    def _prepare_mask(x, mask):
        num_nodes, num_features = x.size()
        device, dtype = x.device, x.dtype

        if mask is None:
            return torch.ones((num_nodes, num_features), device=device, dtype=dtype)

        if not isinstance(mask, torch.Tensor):
            mask = torch.tensor(mask, device=device, dtype=dumpy)

        mask = mask.to(device=device, dtype=dtype)

        if mask.dim() == 1:
            mask = mask.unsqueeze(-1).expand(num_nodes, num_features)

        if mask.dim() == 2 and mask.size(1) == 1:
            mask = mask.expand(num_nodes, num_features)

        return mask


# -------------------------------------------------------------
# PaGNNConv（Normalized 版）
#   H' = ( D_hat ⊙ ( A_hat (M⊙H)  / (A_hat M) ) ) W
# -------------------------------------------------------------
class PaGNNConv(torch.nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.lin = nn.Linear(in_features, out_features)

    def forward(self, x, edge_index, mask):
        num_nodes = x.size(0)
        device = x.device

        # -----------------------------------------
        # 1. 正規化隣接行列 A_hat
        # -----------------------------------------
        edge_index, edge_weight = get_symmetrically_normalized_adjacency(edge_index, num_nodes)
        A_hat = torch.sparse_coo_tensor(
            edge_index,
            edge_weight,
            size=(num_nodes, num_nodes),
            device=device,
        )

        # -----------------------------------------
        # 2. M ⊙ H
        # -----------------------------------------
        X_obs = mask * x

        # -----------------------------------------
        # 3. 分子：A_hat (M ⊙ H)
        # -----------------------------------------
        numerator = torch.sparse.mm(A_hat, X_obs)   # shape: [N, F]

        # -----------------------------------------
        # 4. 分母：A_hat M
        # -----------------------------------------
        denominator = torch.sparse.mm(A_hat, mask)  # shape: [N, F]

        # -----------------------------------------
        # 5. Hadamard division：部分平均
        # -----------------------------------------
        partial = numerator / (denominator + 1e-10)

        # -----------------------------------------
        # 6. D_hat（正規化次数）を掛ける
        #    D_hat[i] = sum_j A_hat[i,j]
        # -----------------------------------------
        D_hat = torch.sparse.sum(A_hat, dim=1).to_dense().unsqueeze(1)
        partial = D_hat * partial

        # -----------------------------------------
        # 7. 線形層 → 出力
        # -----------------------------------------
        out = self.lin(partial)
        return out
