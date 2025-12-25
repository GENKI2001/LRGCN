#!/usr/bin/env python3
"""
CSVファイルから最良結果（valカラムで最高値）のコマンドを抽出してtxtファイルに出力
"""
import argparse
import csv
import re
import sys
from pathlib import Path
from typing import Dict, Optional


def parse_accuracy(value: str) -> float:
    """精度の文字列から中央値を抽出"""
    if not value or str(value).strip() == '':
        return float('-inf')
    s = str(value).strip()
    s = s.lstrip('\ufeff')
    try:
        return float(s.split('±', 1)[0])
    except ValueError:
        return float('-inf')


def get_value(row: Dict[str, str], key: str, default: str = '') -> str:
    """CSV行から値を取得（空文字列の場合はデフォルト値を使用）"""
    val = row.get(key, default)
    if val and str(val).strip():
        return str(val).strip()
    return default


def find_original_n_runs(csv_path: Path, row: Dict[str, str], root: Path) -> int:
    """元のジョブファイルからn_runsを抽出（見つからない場合はデフォルト値から推測）"""
    # CSVファイルのパスから対応するジョブファイルを見つける
    # jobs/{dataset}/{model}/{file_tag}.csv -> jobs/{dataset}/{model}/{file_tag}.txt
    csv_path_str = str(csv_path)
    if 'jobs/' in csv_path_str:
        job_file_path = csv_path_str.replace('.csv', '.txt')
        job_file = Path(job_file_path)
        
        if job_file.exists():
            # ジョブファイルから該当するコマンドを見つける
            # CSVの行のパラメータと一致するコマンドを探す
            dataset = get_value(row, 'dataset', '')
            model = get_value(row, 'model', '')
            label_noise_rate = get_value(row, 'label_noise_rate', '0')
            feature_missing_rate = get_value(row, 'feature_missing_rate', '0')
            feature_noise_rate = get_value(row, 'feature_noise_rate', '0')
            hidden_channels = get_value(row, 'hidden_channels', '128')
            num_layers = get_value(row, 'num_layers', '2')
            
            try:
                with job_file.open('r', encoding='utf-8') as f:
                    for line in f:
                        if (f'--dataset_name {dataset}' in line and 
                            f'--model {model}' in line and
                            f'--label_noise_rate {label_noise_rate}' in line and
                            f'--feature_missing_rate {feature_missing_rate}' in line and
                            f'--feature_noise_rate {feature_noise_rate}' in line and
                            f'--hidden_channels {hidden_channels}' in line and
                            f'--num_layers {num_layers}' in line):
                            # n_runsを抽出
                            match = re.search(r'--n_runs\s+(\d+)', line)
                            if match:
                                return int(match.group(1))
            except Exception:
                pass
    
    # 見つからない場合は、データセットから推測（Texas/Cornell/Wisconsinなら20、それ以外は5）
    dataset = get_value(row, 'dataset', '')
    if dataset in ['Texas', 'Cornell', 'Wisconsin']:
        return 20
    else:
        return 5


def build_command_from_csv_row(row: Dict[str, str], csv_path: Path, root: Path) -> str:
    """CSVの行からpython main.pyコマンドを再構築"""
    model = get_value(row, 'model', '')
    
    # 元のn_runsを取得して5倍にする
    original_n_runs = find_original_n_runs(csv_path, row, root)
    n_runs_5x = original_n_runs * 5
    
    cmd_parts = [
        "python main.py",
        f"--dataset_name {get_value(row, 'dataset', '')}",
        f"--model {model}",
        f"--label_noise_type {get_value(row, 'label_noise_type', 'uniform')}",
        f"--label_noise_rate {get_value(row, 'label_noise_rate', '0')}",
        f"--feature_missing_type {get_value(row, 'feature_missing_type', 'uniform')}",
        f"--feature_missing_rate {get_value(row, 'feature_missing_rate', '0')}",
        f"--feature_noise_rate {get_value(row, 'feature_noise_rate', '0')}",
        f"--hidden_channels {get_value(row, 'hidden_channels', '128')}",
        f"--num_layers {get_value(row, 'num_layers', '2')}",
        f"--lr {get_value(row, 'lr', '0.01')}",
        f"--weight_decay {get_value(row, 'weight_decay', '5e-4')}",
        f"--dropout {get_value(row, 'dropout', '0.5')}",
        f"--epochs {get_value(row, 'epochs', '1000')}",
    ]
    
    # LRGCN固有のパラメータ
    if model == 'LRGCN':
        if get_value(row, 'label_max_hops'):
            cmd_parts.append(f"--label_max_hops {get_value(row, 'label_max_hops', '1')}")
        if get_value(row, 'label_temperature'):
            cmd_parts.append(f"--label_temperature {get_value(row, 'label_temperature', '0.25')}")
    
    # Goodie固有のパラメータ
    if model == 'Goodie':
        if get_value(row, 'lp_alpha'):
            cmd_parts.append(f"--lp_alpha {get_value(row, 'lp_alpha', '0.99')}")
        if get_value(row, 'goodie_lamb'):
            cmd_parts.append(f"--goodie_lamb {get_value(row, 'goodie_lamb', '1')}")
    
    # LP固有のパラメータ（CSVには含まれていないが、デフォルト値を使用）
    if model == 'LP':
        cmd_parts.append(f"--lp_alpha_lp {get_value(row, 'lp_alpha_lp', '0.9')}")
    
    # FISF固有のパラメータ（CSVには含まれていないが、デフォルト値を使用）
    if model == 'FISF':
        cmd_parts.append(f"--fisf_num_iterations {get_value(row, 'fisf_num_iterations', '50')}")
        cmd_parts.append(f"--fisf_alpha {get_value(row, 'fisf_alpha', '0.9')}")
        cmd_parts.append(f"--fisf_beta {get_value(row, 'fisf_beta', '0.7')}")
        cmd_parts.append(f"--fisf_gamma {get_value(row, 'fisf_gamma', '0.1')}")
        cmd_parts.append(f"--fisf_mask_type {get_value(row, 'fisf_mask_type', 'uniform')}")
    
    # n_runsを5倍にして追加
    cmd_parts.append(f"--n_runs {n_runs_5x}")
    
    # results_csvのパスを構築（jobsをexperimentsに置換）
    dataset = get_value(row, 'dataset', '')
    csv_filename = csv_path.name
    
    # CSVファイルのパス構造からexperimentsパスを構築
    # csv_pathが jobs/Cora/LP/file.csv の場合 -> experiments/Cora/LP/file.csv
    csv_path_str = str(csv_path)
    if 'jobs/' in csv_path_str:
        # jobs/以降のパスを取得してexperiments/に置換
        jobs_index = csv_path_str.find('jobs/')
        if jobs_index != -1:
            path_after_jobs = csv_path_str[jobs_index + len('jobs/'):]
            results_csv_path = f"experiments/{path_after_jobs}"
        else:
            results_csv_path = f"experiments/{dataset}/{model}/{csv_filename}"
    else:
        # jobs/が含まれていない場合は、datasetとmodelから構築
        results_csv_path = f"experiments/{dataset}/{model}/{csv_filename}"
    
    cmd_parts.append(f"--results_csv {results_csv_path}")
    
    return " ".join(cmd_parts)


def main():
    ap = argparse.ArgumentParser(description="CSVファイルから最良結果のコマンドを抽出（全体で1つ）")
    ap.add_argument("target_dir", help="CSVを探す親ディレクトリ")
    ap.add_argument("--output", default="best_command.txt", help="出力ファイル名（デフォルト: best_command.txt）")
    ap.add_argument("--encoding", default="utf-8", help="CSVエンコーディング")
    args = ap.parse_args()
    
    root = Path(args.target_dir)
    if not root.is_dir():
        print(f"ERROR: ディレクトリが見つかりません: {root}", file=sys.stderr)
        sys.exit(1)
    
    # CSVファイルを再帰的に検索
    csv_files = sorted(root.rglob("*.csv"))
    
    if not csv_files:
        print(f"INFO: CSVファイルが見つかりませんでした: {root}", file=sys.stderr)
        sys.exit(0)
    
    print(f"INFO: {len(csv_files)} 個のCSVファイルを処理中...")
    
    # すべてのCSVファイルから最良結果を収集
    all_best_rows = []
    for csv_path in csv_files:
        try:
            with csv_path.open('r', encoding=args.encoding, newline='') as f:
                reader = csv.DictReader(f)
                rows = list(reader)
            
            if not rows:
                continue
            
            # valカラムでソート（降順）
            rows_sorted = sorted(
                rows,
                key=lambda r: parse_accuracy(r.get('val', '')),
                reverse=True
            )
            
            # 最良結果を保存（CSVパス情報も含める）
            best_row = rows_sorted[0]
            best_row['_csv_path'] = str(csv_path.relative_to(root))
            best_row['_csv_full_path'] = csv_path  # フルパスも保存
            all_best_rows.append(best_row)
        
        except Exception as e:
            print(f"ERROR: {csv_path} -> {e}", file=sys.stderr)
            continue
    
    if not all_best_rows:
        print("INFO: 有効な結果が見つかりませんでした。", file=sys.stderr)
        sys.exit(0)
    
    # 全体で最良の結果を1つだけ選択
    global_best = max(
        all_best_rows,
        key=lambda r: parse_accuracy(r.get('val', ''))
    )
    
    # コマンドを構築
    csv_path = Path(global_best.get('_csv_full_path', ''))
    best_cmd = build_command_from_csv_row(global_best, csv_path, root)
    
    # 出力ファイルに書き込み（TARGET_DIR直下）
    output_path = root / args.output
    with output_path.open('w', encoding='utf-8') as f:
        f.write(f"# Best command (val accuracy: {global_best.get('val', 'N/A')})\n")
        f.write(f"# Source: {global_best.get('_csv_path', 'unknown')}\n")
        f.write(f"# Dataset: {global_best.get('dataset', 'N/A')}, Model: {global_best.get('model', 'N/A')}\n")
        f.write("\n")
        f.write(best_cmd + "\n")
    
    print(f"完了: 最良コマンドを {output_path} に出力しました。")
    print(f"  Dataset: {global_best.get('dataset', 'N/A')}, Model: {global_best.get('model', 'N/A')}, Val: {global_best.get('val', 'N/A')}")


if __name__ == "__main__":
    main()

