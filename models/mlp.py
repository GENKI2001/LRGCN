import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MLP as PyGMLP


class MLP(nn.Module):
    """
    多層MLPモデル
    グラフ構造を考慮せず、ノード特徴量のみを使用して分類を行う
    """
    
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2, dropout=0.5):
        """
        Args:
            in_channels (int): 入力特徴量の次元
            hidden_channels (int): 隠れ層の次元
            out_channels (int): 出力特徴量の次元
            num_layers (int): レイヤー数（最低2層）
            dropout (float): ドロップアウト率
        """
        super(MLP, self).__init__()
        
        # 最低2層は必要（入力層 + 出力層）
        num_layers = max(2, num_layers)
        
        # レイヤーを構築
        layers = []
        
        # 入力層
        layers.append(torch.nn.Linear(in_channels, hidden_channels))
        layers.append(torch.nn.ReLU())
        if dropout > 0:
            layers.append(torch.nn.Dropout(dropout))
        
        # 中間層（num_layers > 2の場合）
        for _ in range(num_layers - 2):
            layers.append(torch.nn.Linear(hidden_channels, hidden_channels))
            layers.append(torch.nn.ReLU())
            if dropout > 0:
                layers.append(torch.nn.Dropout(dropout))
        
        # 出力層
        layers.append(torch.nn.Linear(hidden_channels, out_channels))
        
        # Sequentialでまとめる
        self.mlp = torch.nn.Sequential(*layers)
        self.num_layers = num_layers
    
    def forward(self, x, edge_index=None, adj=None, adj2=None, edge_weight=None, x_label=None, **kwargs):
        """
        順伝播
        
        Args:
            x (torch.Tensor): ノード特徴量 [num_nodes, in_channels]
            edge_index (torch.Tensor, optional): エッジインデックス（このモデルでは使用されない）
            adj (torch.Tensor, optional): 隣接行列（このモデルでは使用されない）
        
        Returns:
            torch.Tensor: 出力特徴量 [num_nodes, out_channels]
        """
        # グラフ構造は無視して、ノード特徴量のみを使用
        return self.mlp(x) 