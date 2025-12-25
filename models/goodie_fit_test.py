# train_goodie.py
# -*- coding: utf-8 -*-
from typing import Dict
import numpy as np
import time
import copy

import torch
import torch.nn.functional as F
from torch import Tensor

from torch_geometric.nn.models import LabelPropagation  # 公式 LP
from tqdm import tqdm

# あなたの実装ファイルからインポート（パスは環境に合わせて調整してください）
# 例: from models.goodie import Goodie
try:
    from .goodie import Goodie  # Goodie 内で GCN / GCNConv を参照している想定
except ImportError:  # スクリプト単体実行時のフォールバック
    from goodie import Goodie


# =========================================================
# Utilities
# =========================================================
def set_seed(seed: int) -> None:
    import random, os
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

@torch.no_grad()
def performance(logits: Tensor, labels: Tensor) -> Dict[str, float]:
    """
    logits: [N, C], labels: [N] (long)
    Returns: {"acc": float, "macro_f1": float}
    """
    pred = logits.argmax(dim=1)
    acc = (pred == labels).float().mean().item()
    C = logits.size(1)
    f1s = []
    for c in range(C):
        tp = ((pred == c) & (labels == c)).sum().item()
        fp = ((pred == c) & (labels != c)).sum().item()
        fn = ((pred != c) & (labels == c)).sum().item()
        prec = tp / (tp + fp + 1e-12)
        rec  = tp / (tp + fn + 1e-12)
        f1   = 0.0 if (prec + rec) == 0 else (2 * prec * rec) / (prec + rec + 1e-12)
        f1s.append(f1)
    macro_f1 = float(np.mean(f1s))
    return {"acc": float(acc), "macro_f1": macro_f1}


@torch.no_grad()
def simple_neighbor_fill(edge_index: Tensor, x: Tensor, observed_mask: Tensor, num_iterations: int = 1) -> Tensor:
    """
    observed_mask: True=観測済み, False=欠損（埋めたい）
    近傍平均で欠損を埋める（観測済みは固定）
    """
    device = x.device
    N, D = x.size()
    row, col = edge_index
    deg = torch.zeros(N, device=device).index_add_(0, row, torch.ones_like(row, dtype=torch.float, device=device))
    deg = deg.clamp(min=1.0)

    cur = x.clone()
    for _ in range(num_iterations):
        agg = torch.zeros_like(cur).index_add_(0, row, cur[col])
        nbr_mean = agg / deg.unsqueeze(1)
        cur = torch.where(observed_mask.unsqueeze(1), cur, nbr_mean)
    return cur


# =========================================================
# Trainer (args を使わず引数制御)
# =========================================================
def run_goodie_training(
    *,
    x: Tensor,                   # [N, Din]
    edge_index: Tensor,          # [2, E]
    labels: Tensor,              # [N] (long)
    train_mask: Tensor,          # [N] bool
    val_mask: Tensor,            # [N] bool
    test_mask: Tensor,           # [N] bool
    observed_feature_mask: Tensor,  # [N] bool (True=観測済み, False=欠損)
    n_classes: int,
    hidden_channels: int = 64,
    lr: float = 1e-2,
    epochs: int = 200,
    patience: int = 50,
    n_runs: int = 5,
    lp_alpha: float = 0.9,
    lp_layers: int = 50,
    lamb: float = 1.0,
    scaled: bool = False,
    fill_iterations: int = 1,
    leaky_alpha: float = 0.01,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    disable_tqdm: bool = False,
    seed_offset: int = 0,
) -> Dict[str, float]:
    """
    戻り値: {"acc_mean", "acc_std", "acc_var", "macroF_mean", "macroF_std", "macroF_var"}
    """
    Din = x.size(1)
    out_channels = n_classes
    edge_weight = None

    acc_all, f1_all, time_all = [], [], []
    train_all, val_all, test_all = [], [], []

    for run_idx in range(n_runs):
        print(f"\n{'='*60}")
        print(f"Run {run_idx + 1}/{n_runs}")
        print(f"{'='*60}")

        # 各runの開始時刻を記録
        run_start_time = time.time()

        set_seed(seed_offset + run_idx)

        # デバイス移動
        x_dev = x.to(device)
        labels_dev = labels.to(device)
        train_mask_dev = train_mask.to(device)
        val_mask_dev = val_mask.to(device)
        test_mask_dev = test_mask.to(device)
        obs_mask_dev = observed_feature_mask.to(device)
        edge_index_dev = edge_index.to(device)

        # 欠損簡易フィリング（観測済みは固定、欠損だけ近傍平均で更新）
        filled = simple_neighbor_fill(edge_index_dev, x_dev, observed_mask=obs_mask_dev, num_iterations=fill_iterations)
        x_dev = torch.where(obs_mask_dev.unsqueeze(1), x_dev, filled)
        x_dev = torch.nan_to_num(x_dev, 0.0)

        # 公式 Label Propagation
        lp = LabelPropagation(num_layers=lp_layers, alpha=lp_alpha).to(device)
        with torch.no_grad():
            lp_output = lp(labels_dev, edge_index_dev, mask=train_mask_dev)  # [N, C]
            pseudo_labels = lp_output.argmax(dim=1)

        # 擬似ラベルの信頼度（Goodie 側は [N] を想定）
        if lamb != 0.0:
            lp_conf = torch.softmax(lp_output, dim=1).max(dim=1)[0]   # [N]
            lp_conf[train_mask_dev] = 1.0
            weight_mask = None if scaled else lp_conf
        else:
            weight_mask = None

        # モデル
        model = Goodie(
            in_channels=Din,
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            leaky_alpha=leaky_alpha,
            scaled=scaled,
            lamb=lamb,
            dropout=0.5,
            num_layers=1,
        ).to(device)
        optim_ = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0)

        acc_vals = []
        test_results = []
        best_metric = -1.0
        best_model_weights = copy.deepcopy(model.state_dict())
        max_idx = 0
        best_test_result = [0.0, 0.0]  # 安全な初期化

        if disable_tqdm:
            loop = range(1, epochs + 1)
        else:
            loop = tqdm(range(1, epochs + 1), desc=f"Goodie Training (Run {run_idx + 1}/{n_runs})")
        for epoch in loop:
            model.train()
            optim_.zero_grad()

            loss_ce, loss_pseudo = model(
                x_dev, lp_output, edge_index_dev, edge_weight,
                labels_dev, pseudo_labels, train_mask_dev, weight_mask=weight_mask
            )
            loss = loss_ce + float(lamb) * loss_pseudo
            loss.backward()
            optim_.step()

            # 検証
            model.eval()
            with torch.no_grad():
                # 中枢ロジック（fp/lp 埋め込み → attention 混合 → classifier3）は Goodie と同一
                fp_embed = model.classifier1(x_dev, edge_index_dev, edge_weight, embed=True)
                lp_embed = model.classifier2(lp_output, edge_index_dev, edge_weight, embed=True)

                fp_ = model.leakyrelu(torch.mm(fp_embed, model.attention))
                lp_ = model.leakyrelu(torch.mm(lp_embed, model.attention))
                values = torch.softmax(torch.cat((fp_, lp_), dim=1), dim=1)

                output_ = (values[:, 0].unsqueeze(1) * fp_embed) + (values[:, 1].unsqueeze(1) * lp_embed)
                output = model.classifier3(output_, edge_index_dev, edge_weight)

                last_output = output.detach().clone()

                train_metrics = performance(output[train_mask_dev], labels_dev[train_mask_dev])
                val_metrics = performance(output[val_mask_dev], labels_dev[val_mask_dev])
                test_metrics = performance(output[test_mask_dev], labels_dev[test_mask_dev])

                acc_vals.append(val_metrics["acc"])

                if best_metric < val_metrics["acc"]:
                    best_metric = val_metrics["acc"]
                    max_idx = len(acc_vals) - 1
                    best_model_weights = copy.deepcopy(model.state_dict())

                test_results.append([test_metrics["acc"], test_metrics["macro_f1"]])
                best_test_result = test_results[max_idx]

            if not disable_tqdm:
                loop.set_postfix({
                    "loss": float(loss.item()),
                    "val": val_metrics["acc"],
                    "test": test_metrics["acc"],
                })

            # Early Stopping
            if len(acc_vals) - 1 - max_idx > patience:
                print(f"Early stopping at epoch {epoch} (val={val_metrics['acc']:.4f})")
                break

        if not disable_tqdm:
            loop.close()

        # 最終テスト用に最良モデルを復元
        model.load_state_dict(best_model_weights)
        model.eval()
        with torch.no_grad():
            # 最良モデルで出力を再計算
            fp_embed = model.classifier1(x_dev, edge_index_dev, edge_weight, embed=True)
            lp_embed = model.classifier2(lp_output, edge_index_dev, edge_weight, embed=True)

            fp_ = model.leakyrelu(torch.mm(fp_embed, model.attention))
            lp_ = model.leakyrelu(torch.mm(lp_embed, model.attention))
            values = torch.softmax(torch.cat((fp_, lp_), dim=1), dim=1)

            output_ = (values[:, 0].unsqueeze(1) * fp_embed) + (values[:, 1].unsqueeze(1) * lp_embed)
            final_output = model.classifier3(output_, edge_index_dev, edge_weight)

            final_train = performance(final_output[train_mask_dev], labels_dev[train_mask_dev])
            final_val = performance(final_output[val_mask_dev], labels_dev[val_mask_dev])
            final_test = performance(final_output[test_mask_dev], labels_dev[test_mask_dev])

        best_test_at_val = best_test_result[0] if test_results else final_test["acc"]
        best_macro_f1 = best_test_result[1] if test_results else final_test["macro_f1"]

        # 各runの終了時刻を記録
        run_end_time = time.time()
        run_time = run_end_time - run_start_time

        print(
            f"Run {run_idx + 1} Final | Train {final_train['acc']:.3f} | Val {final_val['acc']:.3f} | "
            f"Test {final_test['acc']:.3f} | Best@Val Test {best_test_at_val:.3f} | Time {run_time:.2f}s"
        )

        acc_all.append(float(best_test_at_val))
        f1_all.append(float(best_macro_f1))
        time_all.append(float(run_time))
        train_all.append(float(final_train["acc"]))
        val_all.append(float(final_val["acc"]))
        test_all.append(float(final_test["acc"]))

    acc_all_np = np.array(acc_all) if acc_all else np.array([0.0])
    f1_all_np = np.array(f1_all) if f1_all else np.array([0.0])
    time_all_np = np.array(time_all) if time_all else np.array([0.0])
    train_all_np = np.array(train_all) if train_all else np.array([0.0])
    val_all_np = np.array(val_all) if val_all else np.array([0.0])
    test_all_np = np.array(test_all) if test_all else np.array([0.0])

    acc_mean = float(acc_all_np.mean())
    acc_std = float(acc_all_np.std())
    acc_var = float(acc_all_np.var())
    macro_mean = float(f1_all_np.mean())
    macro_std = float(f1_all_np.std())
    macro_var = float(f1_all_np.var())
    time_mean = float(time_all_np.mean())
    time_std = float(time_all_np.std())
    time_var = float(time_all_np.var())
    train_mean = float(train_all_np.mean())
    train_std = float(train_all_np.std())
    train_var = float(train_all_np.var())
    val_mean = float(val_all_np.mean())
    val_std = float(val_all_np.std())
    val_var = float(val_all_np.var())
    test_mean = float(test_all_np.mean())
    test_std = float(test_all_np.std())
    test_var = float(test_all_np.var())

    return {
        "acc_mean": acc_mean,
        "acc_std": acc_std,
        "acc_var": acc_var,
        "macroF_mean": macro_mean,
        "macroF_std": macro_std,
        "macroF_var": macro_var,
        "time_mean": time_mean,
        "time_std": time_std,
        "time_var": time_var,
        "train_mean": train_mean,
        "train_std": train_std,
        "train_var": train_var,
        "val_mean": val_mean,
        "val_std": val_std,
        "val_var": val_var,
        "test_mean": test_mean,
        "test_std": test_std,
        "test_var": test_var,
    }