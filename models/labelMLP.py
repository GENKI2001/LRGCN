import torch
import torch.nn as nn
import torch.nn.functional as F


class LabelMLP(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels,
                 dropout=0.5, num_layers=1, label_in_channels=None, mask=None):
        super().__init__()
        label_in_dim = in_channels if label_in_channels is None else label_in_channels
        
        self.fc1 = nn.Linear(label_in_dim, hidden_channels)
        self.fc2 = nn.Linear(hidden_channels, out_channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index=None, adj=None, adj2=None, x_label=None, edge_weight=None, **kwargs):
        h = self.fc1(x_label)
        h = F.relu(h)
        h = self.dropout(h)
        return self.fc2(h)