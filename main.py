import torch
import argparse
import numpy as np
import time

from models.gcn import GCN
from models.h2gcn import H2GCN
from models.lrgcn import LRGCN
from models.gcnmf import GCNmf
from models.pagnn import PaGNN
from models.mlp import MLP
from models.lp import LP
from models.fisf import fisf
from models.goodie_fit_test import run_goodie_training
from utils import load_dataset, str_to_bool, write_results_to_csv
from models.fit_test import fit, test
from utils.preprocessing import preprocess_data
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser(description="Train GCN on Planetoid/WebKB/Amazon/Wikipedia datasets with optional label noise and feature missingness.")
    # ラベルノイズ
    parser.add_argument("--label_noise_type", type=str, default="uniform", choices=["uniform", "pair"], help="Label noise type.")
    parser.add_argument("--label_noise_rate", type=float, default=0., choices=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], help="Label noise rate [0.0, 1.0].")
    # 特徴量欠損
    parser.add_argument("--feature_missing_rate", type=float, default=0.9, choices=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], help="Feature missing rate [0.0, 1.0].")
    parser.add_argument("--feature_missing_type", type=str, default="uniform", choices=["uniform", "bias", "struct"], help="Feature missingness type.")
    # 特徴量ガウシアンノイズ
    parser.add_argument("--feature_noise_rate", type=float, default=0.0, choices=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], help="Gaussian feature noise level [0.0, 1.0] (ratio of nodes to corrupt).")
    # エッジノイズ
    parser.add_argument("--edge_noise_rate", type=float, default=0.0, choices=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], help="Edge noise rate [0.0, 1.0].")
    parser.add_argument("--edge_noise_type", type=str, default="flip", choices=["flip", "delete", "add"], help="Edge noise type: flip (delete+add), delete (delete only), add (add only).")
    # 学習設定
    parser.add_argument("--epochs", type=int, default=1000, help="Training epochs.")
    parser.add_argument("--early_stopping", type=str_to_bool, default=True, nargs='?', const=True, help="Use early stopping. Can be True/False or just flag.")
    parser.add_argument("--patience", type=int, default=100, help="Patience for early stopping.")
    parser.add_argument("--n_runs", type=int, default=1, help="Number of experimental runs.")
    # データセット
    parser.add_argument("--dataset_name", type=str, default="Wisconsin", help="Dataset: Cora, Citeseer, Pubmed, Photo, Computers, Texas, Cornell, Wisconsin, Chameleon, Squirrel, Actor")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate.")
    parser.add_argument("--weight_decay", type=float, default=5e-4, help="Weight decay.")
    parser.add_argument("--hidden_channels", type=int, default=128, choices=[32, 64, 128], help="Hidden channels for GCN/H2GCN.")
    parser.add_argument("--num_layers", type=int, default=2, choices=[1, 2], help="Number of GCN layers (>=1).")
    parser.add_argument("--dropout", type=float, default=0.5, help="Dropout rate for hidden layers.")
    # モデル
    parser.add_argument("--model", type=str, default="LRGCN", choices=["GCN", "FISF", "H2GCN", "LRGCN", "GCNmf", "Goodie", "PaGNN", "MLP", "LP"], help="Model to use.")
    # ラベル特徴量
    parser.add_argument("--label_max_hops", type=int, default=1, choices=[1, 2, 3, 4], help="Max hops for label features.")
    parser.add_argument("--label_temperature", type=float, default=2.0, choices=[0.125, 0.25, 0.5, 1.0, 2.0], help="Softmax temperature for label features.")
    # LP固有のオプション
    parser.add_argument("--lp_alpha_lp", type=float, default=0.9, choices=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99, 0.999], help="Label propagation alpha for LP model.")
    # Goodie固有のオプション
    parser.add_argument("--lp_alpha", type=float, default=0.99, choices=[0.8, 0.9, 0.99, 0.999], help="Label propagation alpha for Goodie.")
    parser.add_argument("--goodie_lamb", type=float, default=1.0, choices=[1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10], help="Pseudo-label consistency weight for Goodie.")
    # FISF固有のオプション
    parser.add_argument("--fisf_num_iterations", type=int, default=50, help="Number of iterations for FISF preprocessing.")
    parser.add_argument("--fisf_alpha", type=float, default=0.9, help="Alpha parameter for FISF.")
    parser.add_argument("--fisf_beta", type=float, default=0.7, help="Beta parameter for FISF.")
    parser.add_argument("--fisf_gamma", type=float, default=0.1, help="Gamma parameter for FISF (ratio between 0.0 and 1.0).")
    parser.add_argument("--fisf_mask_type", type=str, default="uniform", choices=["uniform", "structural"], help="Mask type for FISF.")
    # 結果保存ファイル
    parser.add_argument("--results_csv", type=str, default=None,
                        help="Append results as a CSV row to this file.")
    # tqdm進捗バーの無効化
    parser.add_argument("--disable_tqdm", type=str_to_bool, default=False, nargs='?', const=True, help="Disable tqdm progress bar. Can be True/False or just flag.")

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # データセットの読み込み
    dataset, canonical_name = load_dataset(args.dataset_name)
    args.dataset_name = canonical_name

    base_data = dataset[0].clone()

    if args.model == "Goodie":
        results_per_run = []
        for run in range(args.n_runs):
            data = base_data.clone().to(device)
            data, _ = preprocess_data(data, dataset, args, device, random_seed=run)
            data = data.to(device)

            if hasattr(data, "missing_mask"):
                observed_feature_mask = ~data.missing_mask.any(dim=1)
            else:
                observed_feature_mask = torch.ones(data.num_nodes, dtype=torch.bool, device=data.x.device)
            observed_feature_mask = observed_feature_mask.to(torch.bool)

            run_results = run_goodie_training(
                x=data.x,
                edge_index=data.edge_index,
                labels=data.y,
                train_mask=data.train_mask,
                val_mask=data.val_mask,
                test_mask=data.test_mask,
                observed_feature_mask=observed_feature_mask,
                n_classes=dataset.num_classes,
                hidden_channels=args.hidden_channels,
                lr=args.lr,
                epochs=args.epochs,
                patience=args.patience,
                n_runs=1,
                lp_alpha=args.lp_alpha,
                lp_layers=50,
                lamb=args.goodie_lamb,
                scaled=False,
                fill_iterations=1,
                device=str(device),
                disable_tqdm=args.disable_tqdm,
                seed_offset=run,
            )
            results_per_run.append(run_results)

        train_scores = np.array([res["train_mean"] for res in results_per_run])
        val_scores = np.array([res["val_mean"] for res in results_per_run])
        test_scores = np.array([res["test_mean"] for res in results_per_run])
        acc_scores = np.array([res["acc_mean"] for res in results_per_run])
        macro_scores = np.array([res["macroF_mean"] for res in results_per_run])
        time_scores = np.array([res["time_mean"] for res in results_per_run])

        aggregated_results = {
            "train_mean": float(train_scores.mean()),
            "train_std": float(train_scores.std()),
            "val_mean": float(val_scores.mean()),
            "val_std": float(val_scores.std()),
            "test_mean": float(test_scores.mean()),
            "test_std": float(test_scores.std()),
            "acc_mean": float(acc_scores.mean()),
            "acc_std": float(acc_scores.std()),
            "macroF_mean": float(macro_scores.mean()),
            "macroF_std": float(macro_scores.std()),
            "time_mean": float(time_scores.mean()),
            "time_std": float(time_scores.std()),
        }

        if args.n_runs > 1:
            print(f"\n{'='*60}")
            print(f"Statistics over {args.n_runs} runs:")
            print(f"Train: {aggregated_results['train_mean']:.3f}±{aggregated_results['train_std']:.3f}")
            print(f"Val: {aggregated_results['val_mean']:.3f}±{aggregated_results['val_std']:.3f}")
            print(f"Test: {aggregated_results['test_mean']:.3f}±{aggregated_results['test_std']:.3f}")
            print(f"Best@Val Test: {aggregated_results['acc_mean']:.3f}±{aggregated_results['acc_std']:.3f}")
            print(f"Macro-F1: {aggregated_results['macroF_mean']:.3f}±{aggregated_results['macroF_std']:.3f}")
            print(f"Training Time: {aggregated_results['time_mean']:.2f}±{aggregated_results['time_std']:.2f}s")
            print(f"{'='*60}")
        else:
            print(f"\n{'='*60}")
            print(f"Training Time: {aggregated_results['time_mean']:.2f}s")
            print(f"{'='*60}")

        if args.results_csv is not None:
            csv_results = {
                'train_mean': aggregated_results.get('train_mean', 0.0),
                'train_std': aggregated_results.get('train_std', 0.0),
                'val_mean': aggregated_results.get('val_mean', 0.0),
                'val_std': aggregated_results.get('val_std', 0.0),
                'test_mean': aggregated_results.get('test_mean', 0.0),
                'test_std': aggregated_results.get('test_std', 0.0),
                'best_test_at_val': aggregated_results.get('acc_mean', 0.0),
                'acc_std': aggregated_results.get('acc_std', 0.0),
                'time_mean': aggregated_results.get('time_mean', 0.0),
                'time_std': aggregated_results.get('time_std', 0.0),
            }
            write_results_to_csv(args.results_csv, args, csv_results, args.n_runs)

        return

    # 実験結果を保存するリスト
    import copy
    all_results = []

    for run in range(args.n_runs):
        print(f"\n{'='*60}")
        print(f"Run {run + 1}/{args.n_runs}")
        print(f"{'='*60}")

        # 学習開始時刻を記録
        run_start_time = time.time()

        data = base_data.clone().to(device)
        data, _ = preprocess_data(data, dataset, args, device, random_seed=run)
        data = data.to(device)

        # FISFの場合、FISFを前処理として実行
        if args.model == "FISF":
            # missing_maskがある場合はそれを使ってfeature_maskを作成
            if hasattr(data, "missing_mask"):
                # missing_maskはTrueが欠損、Falseが観測なので、反転してfeature_maskを作成
                feature_mask = (~data.missing_mask).to(dtype=torch.bool)
            else:
                # missing_maskがない場合は全て観測されていると仮定
                feature_mask = torch.ones_like(data.x, dtype=torch.bool)
            
            with torch.no_grad():
                # FISFで特徴量を補完
                data.x = fisf(
                    edge_index=data.edge_index,
                    X=data.x,
                    feature_mask=feature_mask,
                    num_iterations=args.fisf_num_iterations,
                    mask_type=args.fisf_mask_type,
                    alpha=args.fisf_alpha,
                    beta=args.fisf_beta,
                    gamma=args.fisf_gamma,
                )
            print("FISF前処理完了: 特徴量を補完しました")

        # モデルを再初期化（各runで独立）
        if args.model == "GCN" or args.model == "FISF":
            model = GCN(
                in_channels=dataset.num_features,
                hidden_channels=args.hidden_channels,
                out_channels=dataset.num_classes,
                num_layers=args.num_layers,
                dropout=args.dropout,
            ).to(device)
        elif args.model == "H2GCN":
            model = H2GCN(
                in_channels=dataset.num_features,
                hidden_channels=args.hidden_channels,
                out_channels=dataset.num_classes,
                num_layers=args.num_layers,
                dropout=args.dropout,
            ).to(device)
        elif args.model == "LRGCN":
            if hasattr(data, "missing_mask"):
                feature_mask = (~data.missing_mask).to(dtype=data.x.dtype)
            else:
                feature_mask = torch.ones_like(data.x)
            model = LRGCN(
                in_channels=dataset.num_features,
                hidden_channels=args.hidden_channels,
                out_channels=dataset.num_classes,
                num_layers=args.num_layers,
                dropout=args.dropout,
                label_in_channels=dataset.num_classes * args.label_max_hops,
                mask=feature_mask,
            ).to(device)
        elif args.model == "GCNmf":
            model = GCNmf(
                x=data.x,
                adj=data.adj,
                in_features=dataset.num_features,
                hidden_features=args.hidden_channels,
                out_features=dataset.num_classes,
                n_components=5,
                dropout=args.dropout,
                device=device,
            ).to(device)
        elif args.model == "PaGNN":
            if hasattr(data, "missing_mask"):
                feature_mask = (~data.missing_mask).to(dtype=data.x.dtype)
            else:
                feature_mask = torch.ones_like(data.x)

            model = PaGNN(
                in_channels=dataset.num_features,
                hidden_channels=args.hidden_channels,
                out_channels=dataset.num_classes,
                num_layers=args.num_layers,
                dropout=args.dropout,
                mask=feature_mask,
            ).to(device)
        elif args.model == "MLP":
            model = MLP(
                in_channels=dataset.num_features,
                hidden_channels=args.hidden_channels,
                out_channels=dataset.num_classes,
                num_layers=args.num_layers,
                dropout=args.dropout,
            ).to(device)
        elif args.model == "LP":
            model = LP(alpha=args.lp_alpha_lp).to(device)
        else:
            raise ValueError(f"Unsupported model type: {args.model}")

        if args.model == "LP":
            optimizer = None
        else:
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        if args.disable_tqdm:
            loop = range(1, args.epochs + 1)
        else:
            loop = tqdm(range(1, args.epochs + 1), desc=f"Training Epochs (Run {run+1})")
        best_val = 0.0
        best_test_at_val = 0.0
        patience_counter = 0
        best_model_weights = copy.deepcopy(model.state_dict())
        if args.model == "LP":
            run_start_time = time.time()
            logits = model(
                x=data.x,
                edge_index=data.edge_index,
                edge_weight=getattr(data, "edge_weight", None),
                y=data.y,
                train_mask=data.train_mask,
            )
            preds = logits.argmax(dim=1)

            def acc(mask):
                mask = mask.bool()
                denom = int(mask.sum())
                if denom == 0:
                    return 0.0
                return float((preds[mask] == data.y[mask]).sum().item() / denom)

            final = {
                "train": acc(data.train_mask),
                "val": acc(data.val_mask),
                "test": acc(data.test_mask),
            }
            best_val = final["val"]
            best_test_at_val = final["test"]
            run_time = time.time() - run_start_time
            print(
                f"Run {run+1} Final | Train {final['train']:.3f} | Val {final['val']:.3f} | Test {final['test']:.3f} | "
                f"Best@Val Test {best_test_at_val:.3f} | Time {run_time:.2f}s"
            )
            all_results.append({
                "train": final['train'],
                "val": final['val'],
                "test": final['test'],
                "best_test_at_val": best_test_at_val,
                "time": run_time
            })
            continue

        for epoch in loop:
            loss = fit(model, data, optimizer)
            metrics = test(model, data)
            if metrics["val"] > best_val:
                best_val = metrics["val"]
                best_test_at_val = metrics["test"]
                patience_counter = 0
                best_model_weights = copy.deepcopy(model.state_dict())
            else:
                patience_counter += 1
            if not args.disable_tqdm:
                loop.set_postfix({"loss": loss, "val": metrics["val"], "test": metrics["test"]})
            if args.early_stopping and patience_counter >= args.patience:
                print(f"Early stopping at epoch {epoch} (val={metrics['val']:.4f})")
                if hasattr(model, "w_label") and model.w_label is not None:
                    w_label = model.w_label.detach().float().flatten()
                    if w_label.numel() > 0:
                        w_mean = float(w_label.mean().item())
                        w_std = float(w_label.std(unbiased=False).item())
                        w_min = float(w_label.min().item())
                        w_max = float(w_label.max().item())
                        print(
                            "w_label stats -> "
                            f"mean: {w_mean:.4f}, std: {w_std:.4f}, "
                            f"min: {w_min:.4f}, max: {w_max:.4f}"
                        )
                break
        # 最終テスト用に最良モデルを復元
        model.load_state_dict(best_model_weights)
        final = test(model, data)
        
        # 学習終了時刻を記録
        run_end_time = time.time()
        run_time = run_end_time - run_start_time
        
        print(
            f"Run {run+1} Final | Train {final['train']:.3f} | Val {final['val']:.3f} | Test {final['test']:.3f} | "
            f"Best@Val Test {best_test_at_val:.3f} | Time {run_time:.2f}s"
        )
        all_results.append({
            "train": final['train'],
            "val": final['val'],
            "test": final['test'],
            "best_test_at_val": best_test_at_val,
            "time": run_time
        })

    # 平均と分散を計算して出力
    if args.n_runs > 1:
        avg_train = sum(r["train"] for r in all_results) / len(all_results)
        avg_val = sum(r["val"] for r in all_results) / len(all_results)
        avg_test = sum(r["test"] for r in all_results) / len(all_results)
        avg_best_test_at_val = sum(r["best_test_at_val"] for r in all_results) / len(all_results)
        avg_time = sum(r["time"] for r in all_results) / len(all_results)
        
        var_train = np.var([r["train"] for r in all_results])
        var_val = np.var([r["val"] for r in all_results])
        var_test = np.var([r["test"] for r in all_results])
        var_best_test_at_val = np.var([r["best_test_at_val"] for r in all_results])
        var_time = np.var([r["time"] for r in all_results])
        
        print(f"\n{'='*60}")
        print(f"Statistics over {args.n_runs} runs:")
        print(f"Train: {avg_train:.3f}±{np.sqrt(var_train):.3f}")
        print(f"Val: {avg_val:.3f}±{np.sqrt(var_val):.3f}")
        print(f"Test: {avg_test:.3f}±{np.sqrt(var_test):.3f}")
        print(f"Best@Val Test: {avg_best_test_at_val:.3f}±{np.sqrt(var_best_test_at_val):.3f}")
        print(f"Training Time: {avg_time:.2f}±{np.sqrt(var_time):.2f}s")
        print(f"{'='*60}")
    else:
        # 単一実行の場合
        avg_train = all_results[0]["train"]
        avg_val = all_results[0]["val"]
        avg_test = all_results[0]["test"]
        avg_best_test_at_val = all_results[0]["best_test_at_val"]
        avg_time = all_results[0]["time"]
        var_train = var_val = var_test = var_best_test_at_val = var_time = 0.0
        print(f"\n{'='*60}")
        print(f"Training Time: {avg_time:.2f}s")
        print(f"{'='*60}")
    
    # 最終結果をCSVに書き込む（全ての実行完了後に1回のみ）
    if args.results_csv is not None:
        # 結果をCSV書き込み用の形式に変換
        csv_results = {
            'train_mean': avg_train,
            'train_std': np.sqrt(var_train),
            'val_mean': avg_val,
            'val_std': np.sqrt(var_val),
            'test_mean': avg_test,
            'test_std': np.sqrt(var_test),
            'best_test_at_val': avg_best_test_at_val,
            'acc_std': np.sqrt(var_best_test_at_val),
            'time_mean': avg_time,
            'time_std': np.sqrt(var_time),
        }
        write_results_to_csv(args.results_csv, args, csv_results, args.n_runs)


if __name__ == "__main__":
    main()


