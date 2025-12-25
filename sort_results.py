#!/usr/bin/env python3
# sort_results_bulk.py
"""
指定フォルダ配下（再帰）にあるすべての CSV を、指定カラムでソートして**上書き**します。
デフォルトは 'val' カラムを降順（大きい→小さい）。
※testカラムでソートするとデータリークになるため、valカラムを使用します。

使い方:
  python sort_results_bulk.py <target_dir>
  python sort_results_bulk.py <target_dir> --column val --ascending   # 昇順
  python sort_results_bulk.py <target_dir> --glob "*.csv"              # パターン変更
  python sort_results_bulk.py <target_dir> --no-recursive               # 直下のみ
  python sort_results_bulk.py <target_dir> --backup                     # .bak を作成
"""

import argparse
import csv
import sys
from pathlib import Path
from typing import List, Tuple

def parse_accuracy(value: str) -> float:
    """
    精度の文字列から中央値を抽出する
    例: "0.816±0.002" -> 0.816
        "0.816"       -> 0.816
        ""/None/NG    -> -inf（並べ替え時に一番下になる）
    """
    if not value or str(value).strip() == '':
        return float('-inf')
    s = str(value).strip()
    # BOM 対応で先頭の不可視文字を除去
    s = s.lstrip('\ufeff')
    # ± 記号で分割して前半を float 化
    try:
        return float(s.split('±', 1)[0])
    except ValueError:
        return float('-inf')

def sort_csv_in_place(csv_path: Path, column: str, ascending: bool, encoding: str = "utf-8") -> Tuple[bool, str, int]:
    """
    単一 CSV を指定カラムで並び替えて上書き保存する。
    戻り値: (成功/失敗, 失敗理由 or "", 件数)
    """
    try:
        with csv_path.open('r', encoding=encoding, newline='') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            fieldnames = reader.fieldnames or []

        if not rows:
            return (True, "empty file (no rows)", 0)

        if column not in fieldnames:
            return (False, f"column '{column}' not found", 0)

        # 並び替え
        rows_sorted = sorted(
            rows,
            key=lambda r: parse_accuracy(r.get(column, "")),
            reverse=not ascending
        )

        with csv_path.open('w', encoding=encoding, newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows_sorted)

        # トップ5を表示用に返す（件数だけ返す。内容は呼び出し元で読まない）
        return (True, "", len(rows_sorted))

    except Exception as e:
        return (False, str(e), 0)

def main():
    ap = argparse.ArgumentParser(description="ディレクトリ配下の全 CSV を指定カラムでソートして上書き保存")
    ap.add_argument("target_dir", help="CSV を探す親ディレクトリ")
    ap.add_argument("--column", default="val", help="並べ替え対象カラム名（デフォルト: val。testカラムはデータリークのため使用しない）")
    ap.add_argument("--ascending", action="store_true", help="昇順（デフォルトは降順）")
    ap.add_argument("--glob", default="*.csv", help='検索パターン（デフォルト: "*.csv"）')
    ap.add_argument("--no-recursive", action="store_true", help="再帰せず直下のみを対象にする")
    ap.add_argument("--encoding", default="utf-8", help='CSV エンコーディング（例: "utf-8", "utf-8-sig"）')
    ap.add_argument("--backup", action="store_true", help="上書き前に <file>.bak を作成する")
    args = ap.parse_args()

    root = Path(args.target_dir)
    if not root.is_dir():
        print(f"ERROR: ディレクトリが見つかりません: {root}", file=sys.stderr)
        sys.exit(1)

    files: List[Path]
    if args.no_recursive:
        files = sorted(root.glob(args.glob))
    else:
        files = sorted(root.rglob(args.glob))

    if not files:
        print(f"INFO: CSV が見つかりませんでした: {root} (pattern={args.glob}, recursive={'no' if args.no_recursive else 'yes'})")
        sys.exit(0)

    print(f"INFO: 対象 CSV: {len(files)} 件（column='{args.column}', order={'ASC' if args.ascending else 'DESC'}）")

    success = 0
    failed = 0
    for p in files:
        try:
            if args.backup:
                bak = p.with_suffix(p.suffix + ".bak")
                bak.write_bytes(p.read_bytes())

            ok, reason, n = sort_csv_in_place(p, args.column, args.ascending, args.encoding)
            if ok:
                print(f"[OK] {p}  ({n} rows)")
                # 上位5件の概要を出す（軽め）
                try:
                    with p.open('r', encoding=args.encoding, newline='') as f:
                        rdr = csv.DictReader(f)
                        top5 = []
                        for i, row in enumerate(rdr):
                            if i >= 5: break
                            top5.append(row.get(args.column, "N/A"))
                    if top5:
                        print("     Top5:", ", ".join(top5))
                except Exception:
                    pass
                success += 1
            else:
                print(f"[NG] {p} -> {reason}")
                failed += 1
        except Exception as e:
            print(f"[NG] {p} -> {e}")
            failed += 1

    print(f"完了: 成功 {success}, 失敗 {failed}")

if __name__ == "__main__":
    main()
