import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import torch


class GCN(nn.Module):
    """
    シンプルGCN。num_layers=1 もOK。
      - num_layers=1: GCNConv(in -> out) のみ（活性化/Dropoutなし）
      - num_layers>=2: [in->hidden] + [hidden->hidden]*(L-2) + [hidden->out]
                       中間層のみ ReLU + Dropout

    Args:
        in_channels (int)
        hidden_channels (int)
        out_channels (int)
        num_layers (int): >=1
        dropout (float)
    """
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2, dropout=0.5):
        super().__init__()
        assert num_layers >= 1, "num_layers は 1 以上にしてください。"
        self.dropout = dropout
        self.num_layers = num_layers

        self.convs = nn.ModuleList()
        if num_layers == 1:
            # 単層：そのまま in -> out
            self.convs.append(GCNConv(in_channels, out_channels, cached=True, normalize=True))
        else:
            # 多層：1層目
            self.convs.append(GCNConv(in_channels, hidden_channels, cached=True, normalize=True))
            # 中間層
            for _ in range(num_layers - 2):
                self.convs.append(GCNConv(hidden_channels, hidden_channels, cached=True, normalize=True))
            # 最終層
            self.convs.append(GCNConv(hidden_channels, out_channels, cached=True, normalize=True))

    def forward(self, x, edge_index, adj=None, adj2=None, edge_weight=None, x_label=None, **kwargs):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index, edge_weight=edge_weight)
            # 中間層のみ活性化+Dropout（= 最終層以外 & 2層以上のとき）
            if self.num_layers >= 2 and i != len(self.convs) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x
