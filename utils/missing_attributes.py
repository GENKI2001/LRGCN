from copy import deepcopy
import torch


def _rand_like(x, generator=None):
    # x と同じ device/shape/dtype（ただし dtype は float）で一様乱数
    return torch.rand(x.size(), device=x.device, generator=generator)


def generate_mask(features, feature_missing_rate, feature_missing_type, generator=None):
    """
    features: torch.Tensor (N, F)
    feature_missing_rate: float
    feature_missing_type: 'uniform' | 'bias' | 'struct'
    generator: torch.Generator | None
    """
    if feature_missing_type == 'uniform':
        return generate_uniform_mask(features, feature_missing_rate, generator=generator)
    if feature_missing_type == 'bias':
        return generate_bias_mask(features, ratio=feature_missing_rate, generator=generator)
    if feature_missing_type == 'struct':
        return generate_struct_mask(features, feature_missing_rate, generator=generator)
    raise ValueError(f"Missing type {feature_missing_type} is not defined")


def generate_uniform_mask(features, feature_missing_rate, generator=None):
    """
    各要素を独立に feature_missing_rate の確率で欠損（True）
    """
    u = _rand_like(features, generator=generator)
    mask = (u <= feature_missing_rate)
    return mask


def generate_bias_mask(features, ratio, high=0.9, low=0.1, generator=None):
    """
    「特徴次元ごとに」欠損確率を high or low に振り分けるバイアス付き欠損。
    ratio は high を採用する特徴次元の比率（0〜1想定）。

    例: ratio=0.3, high=0.9, low=0.1
        全特徴の約30%は 0.9 の高欠損率、残りは 0.1 の低欠損率。
    """
    ratio = float(ratio)
    ratio = max(0.0, min(1.0, ratio))  # 念のためクリップ

    # 特徴次元ごとに「high を採用するか？」を決める (1, F)
    feat_choice = torch.rand((1, features.size(1)), device=features.device, generator=generator) < ratio
    high_t = torch.tensor(high, device=features.device)
    low_t  = torch.tensor(low,  device=features.device)
    feat_threshold = torch.where(feat_choice, high_t, low_t)  # (1, F)

    # 各要素で乱数を振って、featごとのしきい値で欠損判定
    u = _rand_like(features, generator=generator)
    mask = (u < feat_threshold)  # (N, F)
    return mask


def generate_struct_mask(features, feature_missing_rate, generator=None):
    """
    「ノード単位の構造的欠損」。
    各ノードが feature_missing_rate の確率で「そのノードの全特徴が欠損」になる。
    """
    node_mask = torch.rand((features.size(0), 1), device=features.device, generator=generator) <= feature_missing_rate  # (N,1) bool
    mask = node_mask.expand_as(features)  # (N,F) 参照ベースでメモリ効率よく拡張
    return mask


def apply_mask(features, mask):
    """
    欠損True の位置に NaN を入れる（元テンソルは破壊しない）
    """
    out = features.clone()
    out = out.masked_fill(mask, float('nan'))
    return out, mask


def add_missing_attributes_to_features(data, feature_missing_rate, feature_missing_type='uniform', device='cpu', seed=None):
    """
    data.x に属性欠損を付与し、NaN→0.0 で埋めたテンソルを data.x に戻す。
    欠損マスクは data.missing_mask に保存（True=欠損）。
    元のdataオブジェクトは変更せず、コピーを返します。

    Parameters
    ----------
    data : torch_geometric.data.Data
        グラフデータ
    feature_missing_rate : float
        欠損率 [0.0, 1.0]
    feature_missing_type : str
        欠損タイプ: 'uniform', 'bias', 'struct'
    device : str
        使用デバイス
    seed : int, optional
        乱数シード（再現性のため）

    Returns
    -------
    data : torch_geometric.data.Data
        属性欠損を付与したグラフデータ（コピー）
    """
    # 常にコピーを作成してから変更
    data = deepcopy(data)
    
    if feature_missing_rate <= 0.0:
        return data

    # data.x を指定 device に揃える（必要な場合）
    if str(data.x.device) != str(device):
        data.x = data.x.to(device)

    # 任意のシード（再現性が必要なら指定）
    g = None
    if seed is not None:
        g = torch.Generator(device=device)
        g.manual_seed(int(seed))

    features = data.x  # (N, F) on device
    mask = generate_mask(features, feature_missing_rate, feature_missing_type, generator=g)  # bool, same device

    # NaN を入れた版（学習に直接使うわけではないが、ログ用途に保持したければ保存しても良い）
    features_nan = features.masked_fill(mask, float('nan'))

    # NaN を 0.0 に置換して学習に使う（NaN伝播を防ぐ）
    features_filled = torch.nan_to_num(features_nan, nan=0.0)

    # 反映
    data.x = features_filled
    data.missing_mask = mask  # True=欠損

    # 任意：元の NaN 版も保存したい場合
    # data.x_with_nan = features_nan

    return data

