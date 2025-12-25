import torch
import torch.nn as nn
import torch.nn.functional as F


class GCNLayer(nn.Module):
    def forward(self, adj, x):
        # adj: [N, N] (sparse), x: [N, d]
        return torch.sparse.mm(adj, x)


class LRGCNBranch(nn.Module):
    def __init__(self, in_dim, hidden_dim, dropout=0.5, num_layers=1, max_hops=2):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.max_hops = max_hops

        self.fc1 = nn.Linear(in_dim, hidden_dim, bias=False)
        self.gcn_layers = nn.ModuleList([GCNLayer() for _ in range(max_hops)])

        # h0 (1本) + 各層×各hop
        self.output_dim = (max_hops * num_layers + 1) * hidden_dim

        self.dropout = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(self.output_dim)

    def forward(self, x, adjacency_matrices):
        # adjacency_matrices: {"adj1hop": ..., "adj2hop": ...}
        h0 = self.fc1(x)  # [N, H]
        outputs = [h0]
        h_prev = h0

        for _ in range(self.num_layers):
            for hop in range(self.max_hops):
                adj_key = f"adj{hop+1}hop"
                adj = adjacency_matrices.get(adj_key, None)
                if adj is not None:
                    h_hop = self.gcn_layers[hop](adj, h_prev)  # [N, H]
                    outputs.append(h_hop)
                else:
                    outputs.append(torch.zeros_like(h_prev))

            # シンプル設計：直近 max_hops のうち一番古いものを次層入力に
            h_prev = outputs[-self.max_hops] if len(outputs) > self.max_hops else h0

        h = torch.cat(outputs, dim=1)  # [N, D]
        h = self.dropout(h)
        h = self.ln(h)
        return h


class LRGCN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels,
                 dropout=0.5, num_layers=1, label_in_channels=None, mask=None):
        super().__init__()
        label_in_dim = in_channels if label_in_channels is None else label_in_channels
        self.z_feat = LRGCNBranch(in_channels, hidden_channels, dropout, num_layers)
        self.z_label = LRGCNBranch(label_in_dim, hidden_channels, dropout, num_layers)
        D = self.z_feat.output_dim

        self.gate_logit = nn.Parameter(torch.tensor(0.0))
        self.fc_out = nn.Linear(D, out_channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index, adj, adj2, x_label=None, edge_weight=None, **kwargs):
        device = x.device
        if x_label is None:
            x_label = x
        x_label = x_label.to(device)

        adj_mats = {
            "adj1hop": adj.to(device) if adj is not None else None,
            "adj2hop": adj2.to(device) if adj2 is not None else None,
        }

        h_feat = self.z_feat(x, adj_mats)
        h_label = self.z_label(x_label, adj_mats)

        w_label = torch.sigmoid(self.gate_logit)
        w_feat = 1.0 - w_label

        h_fused = w_feat * h_feat + w_label * h_label
        h_fused = self.dropout(h_fused)

        self.w_label = torch.full((h_feat.size(0), 1), w_label.item(), device=device)
        return self.fc_out(h_fused)