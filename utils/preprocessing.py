import time
from typing import Any, Dict, Tuple

import torch
from torch_geometric.data import Data

from .feature_noise import add_gaussian_feature_noise
from .graph import make_undirected
from .label_features import add_label_features_to_data, create_onehot_labels_for_data
from .label_noise import label_process
from .edge_noise import add_edge_noise
from .masks import create_random_masks
from .missing_attributes import add_missing_attributes_to_features

def preprocess_data(
    data: Data,
    dataset: Any,
    args: Any,
    device: torch.device,
    random_seed=None
) -> Tuple[Data, Dict[str, Any]]:
    """
    Apply dataset preprocessing steps prior to model training.

    Returns:
        Tuple containing the transformed data object and a dictionary with
        optional statistics (e.g., elapsed times or indices affected by noise).
    """
    stats: Dict[str, Any] = {}


    # random_seedを指定して、ランダムマスクの作成を行う
    data = create_random_masks(
        data,
        train_ratio=0.4,
        val_ratio=0.3,
        test_ratio=0.3,
        rng_device="cuda",
        random_seed=random_seed,
    )

    # エッジノイズを先に適用（make_undirectedの前に）
    if getattr(args, "edge_noise_rate", 0.0) > 0.0:
        data, edge_stats = add_edge_noise(
            data,
            noise_rate=args.edge_noise_rate,
            noise_type=getattr(args, "edge_noise_type", "flip"),
            random_seed=random_seed if random_seed is not None else 5,
            device=str(device),
        )
        stats["edge_stats"] = edge_stats

    # エッジノイズ適用後にedge_indexの無向グラフ化と正規化された adj/adj2の作成を行う
    data = make_undirected(data, device)

    if args.feature_noise_rate > 0.0:
        data, noisy_idx = add_gaussian_feature_noise(
            data,
            noise_level=args.feature_noise_rate,
            device=str(device),
            use_data_stats=True,
        )
        stats["noisy_idx"] = noisy_idx

    if args.feature_missing_rate > 0.0:
        data = add_missing_attributes_to_features(
            data,
            feature_missing_rate=args.feature_missing_rate,
            feature_missing_type=args.feature_missing_type,
            device=str(device),
            seed=random_seed if random_seed is not None else None,
        )

    if args.label_noise_rate > 0.0:
        orig_train_labels = data.y[data.train_mask]
        noisy_train_labels, modified_idx = label_process(
            labels=orig_train_labels,
            features=data.x[data.train_mask],
            n_classes=dataset.num_classes,
            label_noise_type=args.label_noise_type,
            label_noise_rate=args.label_noise_rate,
            random_seed=random_seed if random_seed is not None else 5,
            debug=True,
        )
        data.y = data.y.clone()
        data.y[data.train_mask] = noisy_train_labels
        stats["modified_idx"] = modified_idx

    run_start_time = time.time()
    if getattr(args, "model", "").upper() == "LRGCN":
        data = create_onehot_labels_for_data(data, dataset, device)
        data = add_label_features_to_data(
            data,
            device=device,
            max_hops=args.label_max_hops,
            temperature=args.label_temperature,
            label_smoothing=0.0,
        )
    run_end_time = time.time()
    print("Label feature の計算時間：", run_end_time - run_start_time, "s")

    return data, stats

