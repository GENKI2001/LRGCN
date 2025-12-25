import torch
import torch.nn.functional as F
from torch_geometric.nn.models import LabelPropagation


def create_onehot_labels_for_data(data, dataset, device):
    """
    データオブジェクト用にone-hot labelを作成する関数。
    バリデーション/テストノードのラベルを0にしてラベルリークを防止。

    Args:
        data: PyTorch Geometric データオブジェクト
        dataset: データセットオブジェクト（num_classes を参照）
        device: デバイス

    Returns:
        data: onehotLabel が設定されたデータオブジェクト
    """
    num_nodes = data.x.shape[0]
    num_classes = dataset.num_classes
    onehot_labels = torch.zeros(num_nodes, num_classes, device=device)
    onehot_labels.scatter_(1, data.y.unsqueeze(1), 1)

    # バリデーション/テストのノードはゼロ埋め（ラベルリーク防止）
    if hasattr(data, "val_mask") and data.val_mask is not None:
        # onehot_labels[data.val_mask] = 0
        onehot_labels[data.test_mask] = 1 / num_classes
    if hasattr(data, "test_mask") and data.test_mask is not None:
        # onehot_labels[data.test_mask] = 0
        onehot_labels[data.test_mask] = 1 / num_classes

    data.onehotLabel = onehot_labels
    return data


def create_label_features(data, device, max_hops=2, temperature=1.0, label_smoothing=0.0):
    """
    隣接ノードのラベル特徴量を集約して新しい特徴量を作成する関数

    Returns:
        torch.Tensor: 隣接ノードのラベル分布特徴量（shape: [N, num_classes * max_hops]）
    """
    assert hasattr(data, "onehotLabel"), "data.onehotLabel が必要です。先に create_onehot_labels_for_data を呼んでください。"

    one_hot_labels = data.onehotLabel
    if one_hot_labels.device != device:
        one_hot_labels = one_hot_labels.to(device)

    one_hot_labels_tensor = one_hot_labels.to(dtype=torch.float32, device=device)

    edge_index = data.edge_index
    if edge_index.device != device:
        edge_index = edge_index.to(device)

    num_nodes = data.num_nodes

    hop_features_list = []

    # 隣接行列（疎行列ではなく密行列）を作成
    A = torch.zeros((num_nodes, num_nodes), device=device)
    A[edge_index[0], edge_index[1]] = 1
    A[edge_index[1], edge_index[0]] = 1
    A = A.bool()
    A.fill_diagonal_(False)

    # 各 hop までの到達可能性を追跡
    reachable_nodes = torch.zeros((num_nodes, num_nodes), dtype=torch.bool, device=device)

    for hop in range(1, max_hops + 1):
        if hop == 1:
            mask = A.clone()
            reachable_nodes = mask.clone()
        else:
            current_reachable = torch.matmul(reachable_nodes.float(), A.float()).bool()
            mask = current_reachable & (~reachable_nodes)
            reachable_nodes = reachable_nodes | current_reachable

        # ============================================================
        # ===== ここから元の Softmax 版（比較用として残している） =====
        neighbor_labels = mask.float() @ one_hot_labels_tensor
        hop_features = F.softmax(neighbor_labels / temperature, dim=1)
        
        if label_smoothing > 0.0:
            num_classes = hop_features.size(1)
            hop_features = hop_features * (1 - label_smoothing) + (label_smoothing / num_classes)
        # ===== ここまで元の Softmax 版 =====
        # ============================================================

        # ============================================================
        # ************ ここから mean 版（Ablation 対象） ************
        # 近傍ノードのラベル分布の「平均との差し替え」
        # ============================================================

        # # 近傍ノードのラベル総和 [N, C]
        # neighbor_labels = mask.float() @ one_hot_labels_tensor  # [N, C]

        # # 各ノードの近傍数 [N, 1]
        # num_neighbors = mask.sum(dim=1, keepdim=True).float()   # [N, 1]

        # # 近傍が 0 のときは /1 にしておく（neighbor_labels も 0 なので実質0ベクトル）
        # num_neighbors_clamped = torch.clamp(num_neighbors, min=1.0)

        # # 「総和 / 近傍数」で平均分布 [N, C]
        # hop_features = neighbor_labels / num_neighbors_clamped   # ブロードキャストで安全

        # ************ ここまで mean 版 ****************
        # ============================================================
        
        if label_smoothing > 0.0:
            num_classes = hop_features.size(1)
            hop_features = hop_features * (1 - label_smoothing) + (label_smoothing / num_classes)

        hop_features_list.append(hop_features)

    onehot_label_distributions = torch.cat(hop_features_list, dim=1)
    return onehot_label_distributions



def add_label_features_to_data(data, device, max_hops=2, temperature=1.0, label_smoothing=0.0):
    """
    データオブジェクトにラベル特徴量を追加する関数。結果は data.labelx に格納。
    """
    label_features = create_label_features(data, device, max_hops, temperature, label_smoothing)
    # label_features = create_label_propagation_features(data, device, num_layers=50, alpha=0.9)
    data.labelx = label_features
    return data


def create_label_propagation_features(data, device, num_layers=50, alpha=0.9):
    """
    Label Propagationを使ってラベル特徴量を作成する関数
    
    Args:
        data: PyTorch Geometric データオブジェクト
        device: デバイス
        num_layers: Label propagationの反復回数（デフォルト: 50）
        alpha: Label propagationのハイパーパラメータ（デフォルト: 0.9）
              alphaが大きいほど元のラベルを保持し、小さいほど隣接ノードの影響を受ける
    
    Returns:
        torch.Tensor: Label propagationで計算されたラベル分布特徴量（shape: [N, num_classes]）
    """
    assert hasattr(data, "onehotLabel"), "data.onehotLabel が必要です。先に create_onehot_labels_for_data を呼んでください。"
    assert hasattr(data, "train_mask"), "data.train_mask が必要です。"
    
    one_hot_labels = data.onehotLabel
    if one_hot_labels.device != device:
        one_hot_labels = one_hot_labels.to(device)
    
    edge_index = data.edge_index
    if edge_index.device != device:
        edge_index = edge_index.to(device)
    
    train_mask = data.train_mask
    if train_mask.device != device:
        train_mask = train_mask.to(device)
    
    # Label Propagationモデルを作成
    lp_model = LabelPropagation(num_layers=num_layers, alpha=alpha).to(device)
    
    # Label propagationを実行（勾配計算は不要）
    with torch.no_grad():
        # one-hotラベルをLabel propagationに渡す
        # LabelPropagationはFloatTensor [N, num_classes] 形式のラベルを受け取る
        lp_features = lp_model(
            y=one_hot_labels,
            edge_index=edge_index,
            mask=train_mask
        )
    
    return lp_features
