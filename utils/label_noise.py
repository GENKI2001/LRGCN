import random
from copy import deepcopy

import numpy as np
import torch
from numpy.testing import assert_array_almost_equal


def add_label_noise_to_data(data, label_noise_rate, num_classes, device="cpu"):
    """
    ラベルノイズを追加する関数（Uniform/Symmetric Noise）

    Uniform (Symmetric) ノイズ：
    正解ラベル y* = i を確率 ε で「他の全クラスに一様」にフリップ
    Pr(ỹ = j | y* = i) = ε/(C-1) (j ≠ i)

    Parameters
    ----------
    data : torch_geometric.data.Data
        グラフデータ
    label_noise_rate : float
        ノイズ率 ε (0.0 - 1.0)
    num_classes : int
        クラス数 C
    device : str
        使用デバイス

    Returns
    -------
    data : torch_geometric.data.Data
        ラベルノイズを追加したグラフデータ
    """
    if label_noise_rate <= 0.0:
        return data

    noisy_data = deepcopy(data)
    num_nodes = noisy_data.y.size(0)
    noisy_labels = noisy_data.y.clone()

    for idx in range(num_nodes):
        current_label = int(noisy_data.y[idx].item())
        if torch.rand(1, device=device).item() < label_noise_rate:
            available_labels = list(range(num_classes))
            available_labels.remove(current_label)
            new_label_idx = torch.randint(0, len(available_labels), (1,), device=device)
            noisy_labels[idx] = available_labels[int(new_label_idx.item())]

    noisy_data.y = noisy_labels
    return noisy_data


def setup_seed(seed):
    """乱数シードを固定（Python / NumPy / Torch / CuDNN）"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def uniform_noise_cp(n_classes, label_noise_rate):
    """
    一様ノイズの汚染確率行列 T を生成（行確率）
      T[i,i] = 1 - r
      T[i,j] = r / (C-1)  (j != i)
    """
    C = int(n_classes)
    r = float(label_noise_rate)
    P = (r / (C - 1)) * np.ones((C, C), dtype=np.float64)
    np.fill_diagonal(P, 1.0 - r)
    assert_array_almost_equal(P.sum(axis=1), np.ones(C), decimal=12)
    return P


def pair_noise_cp(n_classes, label_noise_rate):
    """
    ペアワイズノイズ（循環隣接にフリップ）
      T[i,i] = 1 - r
      T[i, (i-1) mod C] = r
    """
    C = int(n_classes)
    r = float(label_noise_rate)
    P = (1.0 - r) * np.eye(C, dtype=np.float64)
    for i in range(C):
        P[i, (i - 1) % C] = r
    assert_array_almost_equal(P.sum(axis=1), np.ones(C), decimal=12)
    return P


def add_instance_independent_label_noise(labels_np, cp, random_seed):
    """
    インスタンス非依存ノイズを付与（行 cp[y,:] から1回サンプル）
      labels_np: (N,) int
      cp: (C,C) row-stochastic
    """
    assert_array_almost_equal(cp.sum(axis=1), np.ones(cp.shape[0]), decimal=12)
    rs = np.random.RandomState(int(random_seed))

    noisy = labels_np.copy()
    for i in range(labels_np.shape[0]):
        y = int(labels_np[i])
        onehot = rs.multinomial(1, cp[y], size=1)[0]
        cls = int(np.argmax(onehot))
        noisy[i] = cls
    return noisy


def label_process(labels, features, n_classes, label_noise_type="uniform", label_noise_rate=0.0, random_seed=5, debug=True):
    """
    ラベルノイズ付与（uniform / pair）
      labels: torch.LongTensor (N,)
      features: 未使用（将来拡張用、保持）
      n_classes: int
      label_noise_type: 'uniform' | 'pair'
      label_noise_rate: float in [0,1]
      random_seed: int
    Returns:
      noisy_train_labels (torch.LongTensor, device=labels.device)
      modified_mask (np.ndarray of indices where label changed)
    """
    setup_seed(random_seed)
    assert 0.0 <= float(label_noise_rate) <= 1.0

    if debug:
        print('----label noise information:------')

    if label_noise_rate == 0.0:
        if debug:
            print('No label noise (rate=0.0)')
        return labels.clone(), np.array([], dtype=np.int64)

    if label_noise_type == 'uniform':
        if debug:
            print('Uniform noise')
        cp = uniform_noise_cp(n_classes, label_noise_rate)
    elif label_noise_type == 'pair':
        if debug:
            print('Pair noise')
        cp = pair_noise_cp(n_classes, label_noise_rate)
    else:
        raise ValueError(f"Unsupported label_noise_type: {label_noise_type!r} (use 'uniform' | 'pair')")

    labels_np = labels.detach().cpu().numpy()
    noisy_labels_np = add_instance_independent_label_noise(labels_np, cp, random_seed)
    noisy_train_labels = torch.tensor(noisy_labels_np, dtype=torch.long, device=labels.device)

    changed = (noisy_labels_np != labels_np)
    modified_mask = np.nonzero(changed)[0]
    if debug:
        actual_noise_rate = changed.mean() if changed.size > 0 else 0.0
        print(f'#Actual noise rate {actual_noise_rate:.2f}')

    return noisy_train_labels, modified_mask

