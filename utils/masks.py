import torch


def create_random_masks(
    data,
    train_ratio=0.6,
    val_ratio=0.2,
    test_ratio=0.2,
    random_seed=None,
    rng_device="cpu",   # "cpu"（再現性重視, デフォ）, "auto", "cuda"
):
    """
    ランダム train/val/test マスクを生成（層化なし）
    rng_device:
      - "cpu": 乱数はCPUで生成（CPU/GPUに関わらず同じ分割）
      - "cuda": 乱数をCUDAで生成（GPU実行時のみ; 速い/CPUとは一致しない可能性）
      - "auto": data.x.device に合わせる（GPUならcuda, それ以外はcpu）
    """
    total_ratio = train_ratio + val_ratio + test_ratio
    if abs(total_ratio - 1.0) > 1e-6:
        raise ValueError(f"ratios sum to {total_ratio}, must be 1.0")

    num_nodes = data.num_nodes
    target_device = getattr(data.x, "device", torch.device("cpu"))

    if rng_device == "auto":
        rng_device = "cuda" if target_device.type == "cuda" else "cpu"

    # 乱数生成器
    g = torch.Generator(device=rng_device)
    if random_seed is not None:
        g.manual_seed(int(random_seed))

    # 乱順列（rng_device 上で作成）
    perm = torch.randperm(num_nodes, generator=g, device=rng_device)

    train_size = int(num_nodes * train_ratio)
    val_size   = int(num_nodes * val_ratio)
    # 残りは test
    train_idx = perm[:train_size]
    val_idx   = perm[train_size:train_size + val_size]
    test_idx  = perm[train_size + val_size:]

    # マスクは data と同じ device に置く
    train_mask = torch.zeros(num_nodes, dtype=torch.bool, device=target_device)
    val_mask   = torch.zeros(num_nodes, dtype=torch.bool, device=target_device)
    test_mask  = torch.zeros(num_nodes, dtype=torch.bool, device=target_device)

    train_mask[train_idx.to(target_device)] = True
    val_mask[val_idx.to(target_device)]     = True
    test_mask[test_idx.to(target_device)]   = True

    data.train_mask = train_mask
    data.val_mask   = val_mask
    data.test_mask  = test_mask

    return data
