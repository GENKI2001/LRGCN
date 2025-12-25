"""
CSV書き込みユーティリティ
実験結果をCSVファイルに記録する共通関数
"""
import os
import csv
import numpy as np
from typing import Dict, Any, Optional


def write_results_to_csv(
    results_csv: str,
    args: Any,
    results: Dict[str, float],
    n_runs: int,
) -> None:
    """
    実験結果をCSVファイルに書き込む
    
    Args:
        results_csv: CSVファイルのパス
        args: argparseの引数オブジェクト
        results: 結果の辞書。以下のキーを含む必要がある:
            - train_mean, train_std (または train)
            - val_mean, val_std (または val)
            - test_mean, test_std (または test)
            - best_test_at_val (または best_test_at_val_mean, best_test_at_val_std)
            - time_mean, time_std (または time)
        n_runs: 実行回数
    """
    header = [
        "model", "dataset", "label_noise_type", "label_noise_rate",
        "feature_missing_type", "feature_missing_rate", "feature_noise_rate",
        "seed", "epochs", "lr", "weight_decay", "hidden_channels", "num_layers", "dropout",
        "label_max_hops", "label_temperature",
        "lp_alpha", "goodie_lamb",
        "train", "val", "test", "best_test_at_val", "time"
    ]

    # seed は固定値1000を使用（make_jobs.pyで使用されている値に合わせる）
    seed_value = getattr(args, 'seed', 1000)

    # 結果を "mean±std" 形式、または単一実行の場合は "mean" 形式でフォーマット
    if n_runs > 1:
        # Goodie形式（mean/stdが別々のキー）に対応
        train_mean = results.get('train_mean', results.get('train', 0.0))
        train_std = results.get('train_std', 0.0)
        val_mean = results.get('val_mean', results.get('val', 0.0))
        val_std = results.get('val_std', 0.0)
        test_mean = results.get('test_mean', results.get('test', 0.0))
        test_std = results.get('test_std', 0.0)
        best_test_at_val_mean = results.get('best_test_at_val', results.get('acc_mean', 0.0))
        best_test_at_val_std = results.get('acc_std', 0.0)
        time_mean = results.get('time_mean', results.get('time', 0.0))
        time_std = results.get('time_std', 0.0)

        train_str = f"{train_mean:.3f}±{train_std:.3f}"
        val_str = f"{val_mean:.3f}±{val_std:.3f}"
        test_str = f"{test_mean:.3f}±{test_std:.3f}"
        best_test_at_val_str = f"{best_test_at_val_mean:.3f}±{best_test_at_val_std:.3f}"
        time_str = f"{time_mean:.2f}±{time_std:.2f}s"
    else:
        # 単一実行の場合
        train_val = results.get('train_mean', results.get('train', 0.0))
        val_val = results.get('val_mean', results.get('val', 0.0))
        test_val = results.get('test_mean', results.get('test', 0.0))
        best_test_at_val_val = results.get('best_test_at_val', results.get('acc_mean', 0.0))
        time_val = results.get('time_mean', results.get('time', 0.0))

        train_str = f"{train_val:.3f}"
        val_str = f"{val_val:.3f}"
        test_str = f"{test_val:.3f}"
        best_test_at_val_str = f"{best_test_at_val_val:.3f}"
        time_str = f"{time_val:.2f}s"

    row = [
        args.model, args.dataset_name, args.label_noise_type, args.label_noise_rate,
        args.feature_missing_type, args.feature_missing_rate, args.feature_noise_rate,
        seed_value, args.epochs, args.lr, args.weight_decay, args.hidden_channels, args.num_layers, args.dropout,
        args.label_max_hops, args.label_temperature,
        args.lp_alpha, args.goodie_lamb,
        train_str, val_str, test_str, best_test_at_val_str, time_str
    ]

    need_header = (not os.path.exists(results_csv)) or (os.path.getsize(results_csv) == 0)
    with open(results_csv, "a", newline="") as f:
        w = csv.writer(f)
        if need_header:
            w.writerow(header)
        w.writerow(row)
