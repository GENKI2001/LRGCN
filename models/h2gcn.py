import torch
import torch.nn as nn


class GCNLayer(nn.Module):
    """
    学習可能なパラメータを持たないグラフ畳み込み層。
    隣接行列とノード特徴量の間でスパース行列乗算を実行します。

    Args:
        adj (torch.Tensor): スパース隣接行列 [N, N]
        x (torch.Tensor): ノード特徴量 [N, F]

    Returns:
        torch.Tensor: 更新されたノード特徴量 [N, F]
    """

    def forward(self, adj, x):
        return torch.sparse.mm(adj, x)


class H2GCN(nn.Module):
    """
    H2GCN: 同質性と異質性に対応するグラフ畳み込みネットワーク

    Args:
        in_channels (int): ノードあたりの入力特徴量の数
        hidden_channels (int): ノードあたりの隠れ特徴量の数
        out_channels (int): ノードあたりの出力特徴量の数（クラス数）
        dropout (float, optional): ドロップアウト確率。デフォルト: 0.5
        num_layers (int, optional): 伝播層の数。デフォルト: 1
    """

    def __init__(self, in_channels, hidden_channels, out_channels, dropout=0.5, num_layers=1):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_channels = hidden_channels

        # Step 1: Initial ego feature embedding
        self.fc1 = nn.Linear(in_channels, hidden_channels, bias=False)

        # Step 2: Shared sparse propagation layers (no weights, no nonlinearity)
        self.gcn_1hop = GCNLayer()
        self.gcn_2hop = GCNLayer()

        # Step 3: Output classifier (concat: h0 + h1_1 + h2_1 + ... + h1_K + h2_K)
        total_concat_dim = (2 * num_layers + 1) * hidden_channels
        self.fc_out = nn.Linear(total_concat_dim, out_channels)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index, adj, adj2, edge_weight=None, x_label=None, **kwargs):
        """
        H2GCNの順伝播

        Args:
            x (torch.Tensor): ノード特徴量 [N, in_channels]
            edge_index (torch.Tensor): エッジリスト [2, E]
            adj (torch.sparse.FloatTensor): 1-hop隣接行列（スパース）[N, N]
            adj2 (torch.sparse.FloatTensor): 2-hop隣接行列（スパース）[N, N]

        Returns:
            torch.Tensor: ノード埋め込み [N, out_channels]
        """
        # 前計算済みの隣接行列を必須入力として受け取る
        if adj is None or adj2 is None:
            raise ValueError("H2GCN.forward requires precomputed adj and adj2. Compute once in main and pass as arguments.")

        # Initial embedding
        h0 = self.fc1(x)  # [N, H]

        outputs = [h0]  # [h0, h1_1, h2_1, ..., h1_K, h2_K]
        h_prev = h0

        for _ in range(self.num_layers):
            h1 = self.gcn_1hop(adj, h_prev)  # 1-hop
            h2 = self.gcn_2hop(adj2, h_prev)  # 2-hop
            outputs.extend([h1, h2])
            h_prev = h1

        # Concatenate all
        h = torch.cat(outputs, dim=1)
        h = self.dropout(h)

        return self.fc_out(h)

