from copy import deepcopy
import numpy as np
import torch


def add_gaussian_feature_noise(
    data,
    noise_level=0.0,
    random_seed=5,
    device="cpu",
    use_data_stats=True,
    mean=None,
    std=None,
):
    """
    ノードの一部（noise_levelの割合）を無作為に選び、特徴ベクトルをガウシアン乱数で置換する。
    元のdataオブジェクトは変更せず、コピーを返します。

    Parameters
    ----------
    data : torch_geometric.data.Data
        グラフデータ
    noise_level : float
        ノイズを入れるノードの割合 [0.0, 1.0]
    random_seed : int
        乱数シード
    device : str
        使用デバイス
    use_data_stats : bool
        Trueの場合、データの平均・標準偏差を使用。Falseの場合、N(0,1)を使用
    mean : torch.Tensor, optional
        カスタム平均値 (1, F) または (F,)
    std : torch.Tensor, optional
        カスタム標準偏差 (1, F) または (F,)

    Returns
    -------
    data : torch_geometric.data.Data
        ガウシアンノイズを追加したグラフデータ（コピー）
    noisy_idx : np.ndarray
        ノイズを入れたノードID
    """
    assert 0.0 <= float(noise_level) <= 1.0
    
    # 常にコピーを作成してから変更
    data = deepcopy(data)
    
    if noise_level == 0.0:
        return data, np.array([], dtype=np.int64)

    x = data.x
    if not torch.is_floating_point(x):
        x = x.float().to(device)
    else:
        x = x.to(device)

    N, F = x.size()
    rs = np.random.RandomState(int(random_seed))
    
    # PyTorchの乱数シードも設定（torch.normalで使用されるため）
    torch.manual_seed(int(random_seed))

    k = int(np.floor(N * float(noise_level)))
    if k == 0:
        return data, np.array([], dtype=np.int64)

    noisy_idx = np.sort(rs.choice(N, size=k, replace=False)).astype(np.int64)

    if mean is not None and std is not None:
        mu = mean.to(device).reshape(1, F).expand(k, F)
        sigma = std.to(device).reshape(1, F).expand(k, F)
    elif use_data_stats:
        mu_1F = x.mean(dim=0)
        sigma_1F = x.std(dim=0, unbiased=False)
        sigma_1F = torch.clamp(sigma_1F, min=1e-6)
        mu = mu_1F.unsqueeze(0).expand(k, F)
        sigma = sigma_1F.unsqueeze(0).expand(k, F)
    else:
        mu = torch.zeros((k, F), device=device)
        sigma = torch.ones((k, F), device=device)

    noisy_block = torch.normal(mean=mu, std=sigma)
    x_noisy = x.clone()
    noisy_idx_t = torch.as_tensor(noisy_idx, device=device, dtype=torch.long)
    x_noisy[noisy_idx_t] = noisy_block

    data.x = x_noisy
    return data, noisy_idx
