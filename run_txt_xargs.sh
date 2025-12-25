#!/usr/bin/env bash
# run_txt_xargs.sh
# 指定ディレクトリ配下の .txt を再帰的に並列実行（GNU parallel 不要）

set -euo pipefail

usage() {
  cat <<'USAGE'
使い方:
  ./run_txt_xargs.sh [TARGET_DIR] [JOBS]

  [TARGET_DIR] : 実行したい .txt がある親ディレクトリ（再帰検索）
                 デフォルト: jobs/Cora
  [JOBS]       : 並列数。未指定/auto の場合は CPU コア数

例:
  ./run_txt_xargs.sh
  ./run_txt_xargs.sh jobs/Cora
  ./run_txt_xargs.sh jobs/Cora 6
USAGE
}

# ヘルプ
if [[ $# -gt 0 ]] && { [[ "${1:-}" == "-h" ]] || [[ "${1:-}" == "--help" ]]; }; then
  usage; exit 0
fi

# 引数解釈（第1引数が数値/autoならJOBS扱い）
if [[ $# -ge 1 ]] && [[ "${1:-}" != "auto" ]] && [[ ! "${1:-}" =~ ^[0-9]+$ ]]; then
  TARGET_DIR="$1"
  JOBS="${2:-auto}"
else
  TARGET_DIR="jobs/Cora"
  JOBS="${1:-auto}"
fi

# 並列数
if [[ "$JOBS" == "auto" ]]; then
  if command -v nproc >/dev/null 2>&1; then
    JOBS="$(nproc)"
  else
    JOBS="$(sysctl -n hw.ncpu 2>/dev/null || echo 4)"
  fi
fi

# 対象確認
if [[ ! -d "$TARGET_DIR" ]]; then
  echo "ERROR: ディレクトリが見つかりません: $TARGET_DIR" >&2
  exit 1
fi

# .txt 有無
if ! find "$TARGET_DIR" -type f -name '*.txt' | read -r _; then
  echo "INFO: .txt が見つかりませんでした: $TARGET_DIR"
  exit 0
fi

# 件数表示
TXT_COUNT=$(find "$TARGET_DIR" -type f -name '*.txt' | wc -l | xargs)
echo "INFO: ${TXT_COUNT} 個の .txt ファイルが見つかりました: $TARGET_DIR"
echo "INFO: 並列数: $JOBS"

# 実行（各行の頭にファイル名タグ付け）
# 失敗が1つでもあれば xargs 全体の終了コードは非0になります
export LC_ALL=C
EXIT_CODE=0
find "$TARGET_DIR" -type f -name '*.txt' -print0 \
| xargs -0 -I{} -P "$JOBS" bash -c '
  f="$1"
  name="$(basename "$f")"
  # 実行結果をタグ付けして出力（stderrも統合）
  if ! ( bash "$f" 2>&1 | sed "s/^/$name | /" ); then
    echo "$name | ERROR: failed" 1>&2
    exit 1
  fi
' _ {} || EXIT_CODE=$?

echo "完了: ${TARGET_DIR} 配下の全ての .txt を並列実行しました。"

# CSVファイルのソートを実行
if [[ $EXIT_CODE -eq 0 ]] || [[ $EXIT_CODE -eq 123 ]]; then
  # xargsが一部失敗しても続行（EXIT_CODE=123は一部失敗時）
  echo ""
  echo "INFO: CSVファイルをソート中..."
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  if [[ -f "$SCRIPT_DIR/sort_results.py" ]]; then
    python3 "$SCRIPT_DIR/sort_results.py" "$TARGET_DIR" || {
      echo "WARNING: CSVファイルのソートに失敗しましたが、続行します。" >&2
    }
  else
    echo "WARNING: sort_results.py が見つかりません: $SCRIPT_DIR/sort_results.py" >&2
  fi
  
  # 最良結果のコマンドを抽出
  echo ""
  echo "INFO: 最良結果のコマンドを抽出中..."
  if [[ -f "$SCRIPT_DIR/extract_best_commands.py" ]]; then
    python3 "$SCRIPT_DIR/extract_best_commands.py" "$TARGET_DIR" || {
      echo "WARNING: 最良コマンドの抽出に失敗しましたが、続行します。" >&2
    }
  else
    echo "WARNING: extract_best_commands.py が見つかりません: $SCRIPT_DIR/extract_best_commands.py" >&2
  fi
fi

exit $EXIT_CODE
