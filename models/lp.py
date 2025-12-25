import torch
import torch.nn as nn
from torch_geometric.nn.models import LabelPropagation


class LP(nn.Module):
    """
    Label Propagation モデル
    グラフ構造と学習用ノードのラベルから、全ノードのラベルスコアを推定する。
    ノード特徴量 x は利用せず、API 互換性のために受け取るのみ。
    """

    DEFAULT_NUM_LAYERS: int = 50
    DEFAULT_ALPHA: float = 0.9

    def __init__(self):
        super(LP, self).__init__()
        self.num_layers = self.DEFAULT_NUM_LAYERS
        self.alpha = self.DEFAULT_ALPHA

        # PyG の LabelPropagation をそのまま内部で使用
        self.lp = LabelPropagation(num_layers=self.num_layers, alpha=self.alpha)

    def forward(
        self,
        x=None,
        edge_index=None,
        adj=None,
        adj2=None,
        edge_weight=None,
        x_label=None,
        y=None,
        train_mask=None,
    ):
        """
        順伝播

        Args:
            x (torch.Tensor, optional):
                ノード特徴量 [num_nodes, in_channels]（未使用, インターフェース維持用）
            edge_index (torch.LongTensor):
                エッジインデックス [2, num_edges]
            adj, adj2, x_label:
                未使用（他モデルとのインターフェース互換のために受け取るだけ）
            edge_weight (torch.Tensor, optional):
                エッジ重み [num_edges]（任意）
            y (torch.Tensor):
                ノードラベル。
                - LongTensor [num_nodes]: 未ラベル部は -1
                - もしくは FloatTensor [num_nodes, num_classes]: one-hot / soft labels
            train_mask (torch.BoolTensor):
                [num_nodes]，学習に用いるラベル付きノードを示すマスク

        Returns:
            torch.Tensor:
                伝播後のラベルスコア [num_nodes, num_classes]
        """
        if edge_index is None:
            raise ValueError("LP model requires `edge_index`.")
        if y is None:
            raise ValueError("LP model requires `y` (labels).")
        if train_mask is None:
            raise ValueError("LP model requires `train_mask`.")

        out = self.lp(
            y=y,
            edge_index=edge_index,
            mask=train_mask,
            edge_weight=edge_weight,
        )
        return out
